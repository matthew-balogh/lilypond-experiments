import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from _utils.constants import RANDOM_SEED
from _utils.data_contextualization import DATASET_NAMES, DATASET_GETTERS
from _utils.export import export_figure

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(".")))
EXPORTS_DIR = f"{BASE_DIR}/_exports/04_anomaly_contextualization"
os.makedirs(EXPORTS_DIR, exist_ok=True)

VERB = True

for i, (dataset_name, dataset_getter) in enumerate(zip(DATASET_NAMES, DATASET_GETTERS(random_seed=RANDOM_SEED))):

    # load dataset

    print(f"   Name: {dataset_name}")
    X_hist_normal, X_hist_suspect, X_hist_abnormal, X_holdout_normal, X_holdout_suspect, X_holdout_abnormal = dataset_getter()


    # scale data

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xs_hist_normal = scaler.fit_transform(X_hist_normal)
    del X_hist_normal


    # embedding

    from _utils.som_embedding import SOM_Embedding
    from _utils.som_hyparams import obtain_som_hyparams

    hyparams = obtain_som_hyparams(Xs_hist_normal, verb=VERB)
    hyparams["num_iteration"] = 19
    embedding = SOM_Embedding(**hyparams, verb=VERB) \
        .fit(Xs_hist_normal)
    som = embedding.som_


    # test convergence

    from _utils.som import plot_som_convergence_over_epochs
    MQEs, TEs = plot_som_convergence_over_epochs(Xs_hist_normal, epoch_min=1, epoch_max=2, step=1, show_fig=False, verb=VERB, **hyparams)
    print("MQE over epochs:", np.round(MQEs, 3))


    # data transformation

    Xs_hist_suspect = scaler.transform(X_hist_suspect)
    Xs_hist_abnormal = scaler.transform(X_hist_abnormal)

    Xs_holdoout_suspect = scaler.transform(X_holdout_suspect)
    Xs_holdout_abnormal = scaler.transform(X_holdout_abnormal)


    # Lilypond

    from lilypond.basin import Basin

    basin = Basin(som, Xs_hist_normal, random_seed=RANDOM_SEED) \
        .prepare()

    coloring_strategy = "distance_map"
    gap = .2

    from matplotlib.colors import LinearSegmentedColormap
    gentle_binary_cmap = LinearSegmentedColormap.from_list("gentle_binary", ["#ffffff", "#6F6F6F"])

    pad_style = {
        "marker": "s",
        "gap": gap,
        "cmap": gentle_binary_cmap
    }

    petal_style = {
        "hide": True,
    }

    rhizome_style = {
        "zorder": 2,
        "marker_start": "^",
        "marker_end": "3",
        "linewidth": 1.8,
        "opacity": .95,
    }

    rhizome_style_bold = {
        **rhizome_style,
        "linewidth": 3,
    }

    figsize = (18, 6)
    fig, axes = plt.subplots(1, 4, figsize=figsize);
    axes = axes.flatten()
    plt.suptitle("Lilypond visuals of Cardio dataset", fontsize=16, y=0.99)

    ## Distance information and BMU pairs

    pond1 = basin.pond() \
        .set_coloring_strategy(coloring_strategy) \
        .discretize_petals(n_bins=5) \
        .flood(below_activations=0) \
        .style_pad(**pad_style) \
        .style_petal(**petal_style) \
        .style_flood(underwater_opacity=.4) \
        .style_rhizome(**rhizome_style);
    
    ax = axes[0]
    pond1.clean_attract() \
        .see_rhizome(mode="all", ax=ax) \
        .observe(return_fig=True, ax=ax,
                 title="All BMU pairs");
    
    # Projections of a group with their BMU pairs

    pad_style = {
        **pad_style,
        "cmap": gentle_binary_cmap,
    }
        
    attract_style = {
        "color": "black",
        "zorder": 21,
        "marker": ".",
        "size_base": 300,
        "opacity": .9,
        "subsample_ratio": None,
    }
    
    ## Historical suspect

    attract_style = {**attract_style, "color": "tomato"}

    pond3 = basin.pond() \
        .set_coloring_strategy(coloring_strategy) \
        .flood(below_activations=0) \
        .style_pad(**pad_style) \
        .style_petal(**petal_style) \
        .style_flood(underwater_opacity=.4) \
        .style_rhizome(**rhizome_style);
    
    ax = axes[1]
    pond3.clean_attract() \
        .see_rhizome(Xs_hist_suspect, mode="all", ax=ax) \
        .attract(Xs_hist_suspect, **attract_style) \
        .observe(return_fig=True, ax=ax,
                 title="Hist. suspect samples (All BMU pairs)");
    
    # Historical abnormal

    attract_style = {**attract_style, "color": "orangered"}

    pond4 = basin.pond() \
        .set_coloring_strategy(coloring_strategy) \
        .flood(below_activations=0) \
        .style_pad(**pad_style) \
        .style_petal(**petal_style) \
        .style_flood(underwater_opacity=.4) \
        .style_rhizome(**rhizome_style);
    
    ax = axes[2]
    pond4.clean_attract() \
        .see_rhizome(Xs_hist_abnormal, mode="all", ax=ax) \
        .attract(Xs_hist_abnormal, **attract_style) \
        .observe(return_fig=True, ax=ax,
                 title="Hist. abnormal samples (All BMU pairs)");
    
    # New abnormal

    attract_style = {**attract_style, "color": "magenta"}

    pond5 = basin.pond() \
        .set_coloring_strategy(coloring_strategy) \
        .flood(below_activations=0) \
        .style_pad(**pad_style) \
        .style_petal(**petal_style) \
        .style_flood(underwater_opacity=.4) \
        .style_rhizome(**rhizome_style);
    
    ax = axes[3]
    pond5.clean_attract() \
        .see_rhizome(Xs_holdout_abnormal, mode="all", ax=ax) \
        .attract(Xs_holdout_abnormal, **attract_style) \
        .observe(return_fig=True, ax=ax,
                 title="New abnormal samples (All BMU pairs)");

    for ax in axes:
        ax.set_aspect('equal')
        ax.axis("off")

    plt.tight_layout()
    export_figure(fig, EXPORTS_DIR, f"04_{dataset_name}_lilypond_01_titled.png", hide_titles=False)
    export_figure(fig, EXPORTS_DIR, f"04_{dataset_name}_lilypond_01.png", hide_titles=True)


    # export basin
    import pickle
    pickle.dump(basin, open(f"{BASE_DIR}/experiments/_exports/basin_{dataset_name}.pkl", "wb"))
