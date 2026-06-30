import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from _utils.export import export_figure

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(".")))
DATASET_DIR = f"{BASE_DIR}/experiments/_datasets"
EXPORTS_DIR = f"{BASE_DIR}/_exports/02_unsupervised_exploration"
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


    # test convergence

    from _utils.som import plot_som_convergence_over_epochs
    MQEs, TEs = plot_som_convergence_over_epochs(X_scaled, epoch_min=1, epoch_max=2, step=1, show_fig=False, verb=VERB, **hyparams)
    print("MQE over epochs:", np.round(MQEs, 3))


    # Lilypond

    from lilypond.basin import Basin
    from _utils.constants import RANDOM_SEED

    basin = Basin(som, X_scaled, random_seed=RANDOM_SEED) \
        .prepare()

    figsize = (9, 4)
    fig, _ = basin.legacy_pond().visualize(figsize=figsize, hold_on=True);

    ## export fig
    export_figure(fig, EXPORTS_DIR, f"02_{dataset_name}_lilypond_00.png")


    figsize = (10, 6)
    coloring_strategy = "distance_map"

    pad_style = {
        "marker": "s",
        "gap": gap,
    }

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize);

    basin.pond() \
        .set_coloring_strategy(coloring_strategy) \
        .flood(below_activations=0) \
        .style_pad(**pad_style) \
        .style_petal(hide=True) \
        .style_flood(underwater_opacity=.4) \
        .observe(return_fig=True, ax=ax0, title="Distance map");

    basin.pond() \
        .set_coloring_strategy(coloring_strategy) \
        .flood(below_activations=0) \
        .discretize_petals(n_bins=5) \
        .style_pad(**pad_style) \
        .style_petal(magnifier=3, width=1.5, size_base=.4) \
        .style_flood(underwater_opacity=.4) \
        .observe(return_fig=True, ax=ax1, title="Distance map + activations");

    for ax in (ax0, ax1):
        ax.set_aspect('equal')
        ax.axis("off")

    plt.suptitle("Enhanced visualization of the SOM representation using $Lilypond$")
    plt.tight_layout()

    # export fig
    export_figure(fig, EXPORTS_DIR, f"02_{dataset_name}_lilypond_01.png")

    fig.delaxes(ax0)
    # export fig
    export_figure(fig, EXPORTS_DIR, f"02_{dataset_name}_lilypond_01_simple.png")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize);

    rhizome_style = {
        "zorder": 11,
        "marker_start": "^",
        "marker_end": "3",
        "opacity": .9,
        "linewidth": .1 if dataset_id == "73_mushroom" else 2.5,
    }

    basin.pond() \
        .set_coloring_strategy(coloring_strategy) \
        .flood(below_activations=0) \
        .style_pad(**pad_style) \
        .style_petal(hide=True) \
        .style_flood(underwater_opacity=.4) \
        .style_rhizome(**rhizome_style) \
        .see_rhizome(mode="all", ax=ax0) \
        .observe(return_fig=True, ax=ax0, title="All BMU pairs");
    
    basin.pond() \
        .set_coloring_strategy(coloring_strategy) \
        .flood(below_activations=0) \
        .style_pad(**pad_style) \
        .style_petal(hide=True) \
        .style_flood(underwater_opacity=.4) \
        .style_rhizome(**rhizome_style) \
        .see_rhizome(mode="violating", ax=ax1) \
        .observe(return_fig=True, ax=ax1, title="Violating BMU pairs");
    
    for ax in (ax0, ax1):
        ax.set_aspect('equal')
        ax.axis("off")

    plt.suptitle("Rhizome layer denoting edges between $1st$ and $2nd$ BMUs of the data points")
    plt.tight_layout()

    # export fig
    export_figure(fig, EXPORTS_DIR, f"02_{dataset_name}_lilypond_02.png")

    fig, ax0 = plt.subplots(1, 1, figsize=figsize);

    pad_style = {
        **pad_style,
        "gap": gap * 1.125,
        "cmap": "RdYlGn_r"
    }

    rhizome_style = {
        **rhizome_style,
        "opacity": .8,
        "linewidth": .2 if dataset_id == "73_mushroom" else 5,
    }

    basin.pond() \
        .set_coloring_strategy(coloring_strategy) \
        .flood(below_activations=0) \
        .style_pad(**pad_style) \
        .style_petal(hide=True) \
        .style_flood(underwater_opacity=.4) \
        .style_rhizome(**rhizome_style) \
        .see_rhizome(mode="all", ax=ax0) \
        .observe(return_fig=True, ax=ax0, title="All BMU pairs");

    ax0.set_aspect('equal')
    ax0.axis("off")

    plt.tight_layout()

    # export fig
    export_figure(fig, EXPORTS_DIR, f"02_{dataset_name}_lilypond_03.png")


    # export basin
    import pickle
    pickle.dump(basin, open(f"{BASE_DIR}/experiments/_exports/basin_{dataset_name}.pkl", "wb"))

