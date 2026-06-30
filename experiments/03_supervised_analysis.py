import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from _utils.export import export_figure

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(".")))
DATASET_DIR = f"{BASE_DIR}/experiments/_datasets"
EXPORTS_DIR = f"{BASE_DIR}/_exports/03_supervised_analysis"
os.makedirs(EXPORTS_DIR, exist_ok=True)

DATASET_IDS = ["53_iris", "42_glass", "73_mushroom"]
DATASET_NAMES = ["Iris", "Glass", "Mushroom"]
GAPS = [0.225, 0.25, 0.26]
EPOCHS = [17, 16, 19]
VERB = True

for i, (dataset_id, gap, epoch) in enumerate(zip(DATASET_IDS, GAPS, EPOCHS)):
    print(f"\n\n{i+1}. Dataset ID: {dataset_id}")


    # load dataset

    dataset = pd.read_csv(f'{DATASET_DIR}/{dataset_id}.data', header=None, index_col=None, na_values='?')
    dataset_name = DATASET_NAMES[i]
    print(f"   Name: {dataset_name}")

    if dataset_id == "53_iris":
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]
    elif dataset_id == "42_glass":
        X = dataset.iloc[:, 1:-1]
        y = dataset.iloc[:, -1]
    elif dataset_id == "73_mushroom":
        X = dataset.iloc[:, 1:]
        y = dataset.iloc[:, 0]


    # encode data

    if dataset_id == "73_mushroom":
        X = pd.get_dummies(X, dummy_na=True)


    # encode labels

    from sklearn.preprocessing import LabelEncoder
    from matplotlib.colors import ListedColormap

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    label_decoder = {i: label for i, label in enumerate(le.classes_)}

    num_unique_y = len(np.unique(y_encoded))
    original_cmap = plt.cm.get_cmap('Set1_r')
    colors = original_cmap(np.linspace(0, 1, num_unique_y))
    cmap = ListedColormap(colors)


    # scale data

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    del X


    # embedding

    from _utils.som_embedding import SOM_Embedding
    from _utils.som_hyparams import obtain_som_hyparams

    hyparams = obtain_som_hyparams(X_scaled, verb=VERB)
    hyparams["num_iteration"] = epoch
    embedding = SOM_Embedding(**hyparams, verb=VERB) \
        .fit(X_scaled)
    som = embedding.som_


    # Lilypond

    from lilypond.basin import Basin
    from _utils.constants import RANDOM_SEED

    basin = Basin(som, X_scaled, random_seed=RANDOM_SEED) \
        .prepare()

    figsize = (10, 6)
    coloring_strategy = "distance_map"

    pad_style = {
        "marker": "s",
        "gap": gap,
    }

    attract_style = {
        "cmap": cmap,
        "cmap_values": y_encoded,
        "cmap_label": "Class",
        "label": dataset_name,
        "zorder": 21,
        "marker": "^",
        "size_base": 150 if dataset_id == 73 else 300,
        "opacity": .95,
        "subsample_ratio": None,
    }

    rhizome_style = {
        "zorder": 11,
        "marker_start": "^",
        "marker_end": "3",
        "opacity": .9,
        "linewidth": .1 if dataset_id == "73_mushroom" else 2.5,
    }

    figsize = (10, 11)
    fig, ax0 = plt.subplots(1, 1, figsize=figsize);

    pad_style = {
        **pad_style,
        "gap": gap * 1.05,
    }

    rhizome_style = {
        **rhizome_style,
        "opacity": .8,
        "linewidth": .3 if dataset_id == "73_mushroom" else 7,
    }

    basin.pond() \
        .set_coloring_strategy(coloring_strategy) \
        .flood(below_activations=0) \
        .style_pad(**pad_style) \
        .style_petal(hide=True) \
        .style_flood(underwater_opacity=.4) \
        .style_rhizome(**rhizome_style) \
        .see_rhizome(mode="all", ax=ax0) \
	    .attract(X_scaled, **attract_style) \
        .observe(return_fig=True, ax=ax0, title="All BMU pairs");

    ax0.set_aspect('equal')
    ax0.axis("off")

    plt.tight_layout()

    # export fig
    export_figure(fig, EXPORTS_DIR, f"03_{dataset_name}_lilypond.png")
