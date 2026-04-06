import networkx as nx
import matplotlib.pyplot as plt
import random
import json
import os

def generate_random_graph(seed):
    random.seed(seed)

    G = nx.Graph()
    nodes = list("ABCDEFGHIJKLMNOPQRSTUVWX")
    G.add_nodes_from(nodes)
    
    shuffled_nodes = nodes[:]
    random.shuffle(shuffled_nodes)

    for i in range(len(shuffled_nodes) - 1):
        u = shuffled_nodes[i]
        v = shuffled_nodes[i + 1]
        G.add_edge(u, v)

    # Add extra random edges
    possible_edges = [
        (u, v)
        for i, u in enumerate(nodes)
        for v in nodes[i+1:]
        if not G.has_edge(u, v)
    ]

    extra_edges = random.sample(possible_edges, k=30)

    for u, v in extra_edges:
        G.add_edge(u, v)

    # Add attributes
    for u, v in G.edges():
        base_violence = random.randint(1, 6)
        extra_day_violence = random.randint(0, 5)

        base_terrain = random.randint(1, 6)
        extra_night_terrain = random.randint(1, 5)

        G[u][v]["violence_night"] = base_violence
        G[u][v]["violence_day"] = base_violence + extra_day_violence
        G[u][v]["terrain_day"] = base_terrain
        G[u][v]["terrain_night"] = base_terrain + extra_night_terrain

    return G

def save_graph_json(G, filename):
    data = nx.node_link_data(G)
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def draw_graph(G, image_filename, title="Random Map"):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)  # fixed for reproducible layout
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=900,
        font_size=8
    )
    plt.title(title)
    plt.savefig(image_filename, bbox_inches="tight")
    plt.close()

for i in range(50):
    G = generate_random_graph(seed=i)

    json_path = f"maps/map_{i}.json"
    image_path = f"map_images/map_{i}.png"

    save_graph_json(G, json_path)
    draw_graph(G, image_path, title=f"Map {i}")

print("Done. Saved 50 maps and 50 images.")