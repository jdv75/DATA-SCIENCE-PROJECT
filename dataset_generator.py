import os
import json
import random
import networkx as nx
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# =========================
# Reproducibility
# =========================
random.seed(44)
np.random.seed(42)

# =========================
# Helper functions
# =========================
def load_graph(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return nx.node_link_graph(data, edges="links")

def compute_weight(attrs, time_of_day='day'):
    if time_of_day == 'day':
        violence = attrs['violence_day']
        terrain = attrs['terrain_day']
    else:
        violence = attrs['violence_night']
        terrain = attrs['terrain_night']

    return round(0.7 * violence + 0.3 * terrain, 2)

def apply_weights(G, time_of_day='day'):
    H = G.copy()
    for u, v, attrs in H.edges(data=True):
        attrs['weight'] = compute_weight(attrs, time_of_day)
    return H

def supplies_needed(path):
    return len(path) - 1

def evaluate_option(G, origin, destination, supplies, time_of_day):
    H = apply_weights(G, time_of_day)
    path = nx.shortest_path(H, origin, destination, weight='weight')
    cost = nx.shortest_path_length(H, origin, destination, weight='weight')
    steps = supplies_needed(path)
    feasible = steps <= supplies

    return {
        "time_of_day": time_of_day,
        "path": path,
        "cost": cost,
        "steps": steps,
        "feasible": feasible
    }

def best_time_to_travel(G, origin, destination, supplies):
    day_result = evaluate_option(G, origin, destination, supplies, 'day')
    night_result = evaluate_option(G, origin, destination, supplies, 'night')

    if day_result["feasible"] and not night_result["feasible"]:
        best = "day"
    elif night_result["feasible"] and not day_result["feasible"]:
        best = "night"
    elif day_result["feasible"] and night_result["feasible"]:
        best = "day" if day_result["cost"] <= night_result["cost"] else "night"
    else:
        best = "impossible"

    return best, day_result, night_result

# =========================
# Dataset generation
# =========================

data = []
maps_folder = "maps"
map_files = sorted([f for f in os.listdir(maps_folder) if f.endswith(".json")])

samples_per_map = 200

for map_idx, map_file in enumerate(map_files):
    G = load_graph(os.path.join(maps_folder, map_file))
    districts = list(G.nodes())

    for _ in range(samples_per_map):
        origin, destination = random.sample(districts, 2)
        supplies = random.randint(1, 12)

        best_option, day_result, night_result = best_time_to_travel(G, origin, destination, supplies)
        unweighted_steps = len(nx.shortest_path(G, origin, destination)) - 1

        data.append({
            "map_id": map_idx,
            "origin": origin,
            "destination": destination,
            "supplies": supplies,

            "day_cost": day_result["cost"],
            "night_cost": night_result["cost"],
            "day_steps": day_result["steps"],
            "night_steps": night_result["steps"],

            "cost_diff": day_result["cost"] - night_result["cost"],
            "step_diff": day_result["steps"] - night_result["steps"],
            "shortest_unweighted_steps": unweighted_steps,
            "supply_margin_day": supplies - day_result["steps"],
            "supply_margin_night": supplies - night_result["steps"],

            "best_option": best_option
        })

df = pd.DataFrame(data)

print("First rows of dataset:")
print(df.head())
print("\nClass distribution:")
print(df["best_option"].value_counts())
print("\nDataset shape:", df.shape)

df.to_csv("military_routes_dataset_50_maps.csv", index=False)
print("\nSaved as military_routes_dataset_50_maps.csv")

# =========================
# Split by map_id
# =========================

map_ids = df["map_id"].unique()
np.random.shuffle(map_ids)

split_idx = int(0.8 * len(map_ids))
train_maps = map_ids[:split_idx]
test_maps = map_ids[split_idx:]

train_df = df[df["map_id"].isin(train_maps)]
test_df = df[df["map_id"].isin(test_maps)]

print("\nTrain shape:", train_df.shape)
print("Test shape:", test_df.shape)

X_train = train_df.drop(columns=["best_option", "map_id"])
y_train = train_df["best_option"]

X_test = test_df.drop(columns=["best_option", "map_id"])
y_test = test_df["best_option"]

# =========================
# Label encoding
# =========================

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# =========================
# Preprocessing
# =========================

categorical_features = ["origin", "destination"]
numerical_features = [
    "supplies",
    "day_cost",
    "night_cost",
    "day_steps",
    "night_steps",
    "cost_diff",
    "step_diff",
    "shortest_unweighted_steps",
    "supply_margin_day",
    "supply_margin_night"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numerical_features)
    ]
)

# =========================
# Models
# =========================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
}

# =========================
# Training and evaluation
# =========================

results = []

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train_encoded)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test_encoded, y_pred)
    f1_macro = f1_score(y_test_encoded, y_pred, average="macro")
    f1_weighted = f1_score(y_test_encoded, y_pred, average="weighted")

    print(f"\n{name}")
    print("Accuracy:", acc)
    print("F1 Macro:", f1_macro)
    print("F1 Weighted:", f1_weighted)
    print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))

    results.append({
        "Model": name,
        "Accuracy": acc,
        "F1 Macro": f1_macro,
        "F1 Weighted": f1_weighted
    })

results_df = pd.DataFrame(results)

print("\nComparison table:")
print(results_df.sort_values(by="F1 Macro", ascending=False))

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

best_model = RandomForestClassifier(n_estimators=200, random_state=42)

best_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", best_model)
])

best_pipeline.fit(X_train, y_train_encoded)
y_pred_best = best_pipeline.predict(X_test)

cm = confusion_matrix(y_test_encoded, y_pred_best)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()

rf_model = best_pipeline.named_steps["classifier"]
importances = rf_model.feature_importances_
feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print(importance_df.head(15))

top_n = 15
top_features = importance_df.head(top_n)

plt.figure(figsize=(10,6))
plt.barh(top_features["Feature"][::-1], top_features["Importance"][::-1])
plt.xlabel("Importance")
plt.title("Top 15 Feature Importances - Random Forest")
plt.tight_layout()
plt.show()


# =========================
# Ablation Study
# =========================

print("\n=========================")
print("ABLATION STUDY")
print("=========================")

# Reduced feature set (remove "oracle-like" features)
reduced_numerical_features = [
    "supplies",
    "cost_diff",
    "step_diff",
    "shortest_unweighted_steps"
]

reduced_preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), reduced_numerical_features)
    ]
)

ablation_results = []

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", reduced_preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train_encoded)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test_encoded, y_pred)
    f1_macro = f1_score(y_test_encoded, y_pred, average="macro")
    f1_weighted = f1_score(y_test_encoded, y_pred, average="weighted")

    print(f"\n{name} (Reduced Features)")
    print("Accuracy:", acc)
    print("F1 Macro:", f1_macro)
    print("F1 Weighted:", f1_weighted)

    ablation_results.append({
        "Model": name,
        "Accuracy": acc,
        "F1 Macro": f1_macro,
        "F1 Weighted": f1_weighted
    })

ablation_df = pd.DataFrame(ablation_results)

print("\nAblation Comparison Table:")
print(ablation_df.sort_values(by="F1 Macro", ascending=False))


# PCA Projection

from sklearn.decomposition import PCA

X_test_transformed = best_pipeline.named_steps["preprocessor"].transform(X_test)
if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()
X_test_transformed = X_test_transformed.astype(float)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test_transformed)

plt.figure(figsize=(8, 6))
label_colors = {"day": "steelblue", "night": "darkorange", "impossible": "crimson"}
for label in label_encoder.classes_:
    mask = y_test.values == label
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label, alpha=0.5, s=12, color=label_colors[label])
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.title("PCA Projection of Test Set (colored by label)")
plt.legend()
plt.tight_layout()
plt.show()

# SHAP Values

import shap

feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
rf_clf = best_pipeline.named_steps["classifier"]

explainer = shap.TreeExplainer(rf_clf)
shap_values = explainer.shap_values(X_test_transformed)

shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, class_names=label_encoder.classes_, plot_type="bar", show=True)

