# Learning Optimal Routing Strategies Under Dynamic Conditions

CS439 Final Project — Juan Vasquez, Hiya Savaliya, Patrick Hwang

## What this is

We simulate a military routing problem using randomly generated graphs. Each graph represents a map where nodes are locations and edges have violence and terrain values that differ between day and night. We generate 10,000 routing scenarios and train machine learning models to predict whether traveling during the day, night, or not at all is optimal.

## Setup

Install dependencies:

```
pip install networkx pandas numpy scikit-learn matplotlib seaborn shap
```

## How to run

**Step 1 — Generate maps** (already done, maps/ folder is included):
```
py Maps_generator.py
```

**Step 2 — Run the ML pipeline:**
```
py dataset_generator.py
```

This will generate `military_routes_dataset_50_maps.csv` and output model results, confusion matrix, feature importance, PCA projection, and SHAP analysis.

## Files

- `Maps_generator.py` — generates 50 random graphs and saves them as JSON and PNG
- `dataset_generator.py` — builds the dataset, trains models, and runs all evaluations
- `maps/` — generated graph files (JSON)
- `map_images/` — visualizations of each map (PNG)
