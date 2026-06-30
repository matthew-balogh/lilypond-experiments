import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from _utils.export import export_figure
from lilypond.basin import Basin
from _utils.constants import RANDOM_SEED

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(".")))
DATASET_DIR = f"{BASE_DIR}/experiments/_datasets"
EXPORTS_DIR = f"{BASE_DIR}/_exports/05_code_comparison"
os.makedirs(EXPORTS_DIR, exist_ok=True)

VERB = True

dataset_id = "73_mushroom"
dataset_name = "Mushroom"
gap = 0.26

pond_cmap = LinearSegmentedColormap.from_list("PondGreens", [
    (0.05, 0.15, 0.05),   # dark
    (0.15, 0.35, 0.15),   # natural
    (0.25, 0.55, 0.25),   # medium seagreen
    (0.35, 0.7, 0.35),    # lighter
    (0.45, 0.85, 0.45)    # subtle soft
], N=256)

print(f"\n\nDataset ID: {dataset_id}")


# load dataset

dataset = pd.read_csv(f'{DATASET_DIR}/{dataset_id}.data', header=None, index_col=None, na_values='?')
print(f"   Name: {dataset_name}")

if dataset_id == "73_mushroom":
    X = dataset.iloc[:, 1:]
    y = dataset.iloc[:, 0]


# encode data

if dataset_id == "73_mushroom":
    X = pd.get_dummies(X, dummy_na=True)

# encode labels

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)
label_decoder = {i: label for i, label in enumerate(le.classes_)}

num_unique_y = len(np.unique(y_encoded))
original_cmap = plt.cm.get_cmap('Set1_r')
colors = original_cmap(np.linspace(0, 1, num_unique_y))
cmap_target = ListedColormap(colors)


# scale data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
del X


# embedding

from _utils.som_embedding import SOM_Embedding
from _utils.som_hyparams import obtain_som_hyparams

hyparams = obtain_som_hyparams(X_scaled, verb=VERB)
embedding = SOM_Embedding(**hyparams, verb=VERB) \
    .fit(X_scaled)
som = embedding.som_

def conventional_plot_method(som, data, target):
    w_x, w_y = zip(*[som.winner(d) for d in data])
    w_x = np.array(w_x)
    w_y = np.array(w_y)

    fig = plt.figure(figsize=(10, 9));
    ax = plt.gca();
    plt.pcolor(som.distance_map(), cmap=pond_cmap, alpha=.9)

    for c in np.unique(target):
        idx_target = target==c
        plt.scatter(w_y[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8,
                    w_x[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8, 
                    s=10, c=cmap_target.colors[c], alpha=.95)

    ax.set_aspect('equal')
    ax.axis("off")

    return fig

def lilypond_plot_method(som, data, target):
    pad_style = {
        "marker": "s",
        "gap": gap,
    }

    petal_style = {"hide": True}

    attract_style = {
        "cmap": cmap_target,
        "cmap_values": target,
        "cmap_label": "Class",
        "label": dataset_name,
        "zorder": 21,
        "marker": ".",
        "size_base": 10,
        "opacity": .95,
        "subsample_ratio": None,
    }

    figsize = (6, 6)
    fig = plt.figure(figsize=figsize);
    ax = plt.gca();

    basin = Basin(som, data, random_seed=RANDOM_SEED).prepare();
    basin.pond() \
        .set_coloring_strategy(strategy="distance_map") \
        .flood(below_activations=0) \
        .style_pad(**pad_style) \
        .style_petal(**petal_style) \
	    .attract(data, **attract_style) \
        .observe(return_fig=True, ax=ax, title="");
    
    ax.set_aspect('equal')
    ax.axis("off")

    return fig


fig = conventional_plot_method(som, X_scaled, y_encoded);
export_figure(fig, EXPORTS_DIR, f"05_{dataset_name}_conventional_01.png")

fig = lilypond_plot_method(som, X_scaled, y_encoded);
all_axes = fig.get_axes()
colorbar_axis = all_axes[-1]
colorbar_axis.remove()
export_figure(fig, EXPORTS_DIR, f"05_{dataset_name}_lilypond_01.png")
