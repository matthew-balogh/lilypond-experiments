import numpy as np
import matplotlib.pyplot as plt

from minisom import MiniSom

def train_som(X, d1, d2, sigma, topology, learning_rate, num_iteration,
			  decay_function, sigma_decay_function, use_epochs, random_order,
			  initial_weights=None, random_seed=None, verb=True):
	
	som = MiniSom(d1, d2, input_len=X.shape[1], sigma=sigma, learning_rate=learning_rate, topology=topology,
			   decay_function=decay_function, sigma_decay_function=sigma_decay_function, random_seed=random_seed)

	if initial_weights is None:
		som.random_weights_init(X)
	else:
		som._weights = initial_weights

	som.train(X, num_iteration=num_iteration, random_order=random_order, use_epochs=use_epochs, verbose=verb)

	if verb:
		QE, TE, QE_ROUNDED, TE_ROUNDED = calc_som_main_qualities(som, X)
		print("\nBrief quality of SOM:")
		print(f"Quantization error:\t{QE}")
		print(f"Topographic error:\t{TE}")
		print(f"Quantization error (rounded):\t{QE_ROUNDED}")
		print(f"Topographic error (rounded):\t{TE_ROUNDED}")

	return som

def calc_som_main_qualities(som:MiniSom, X, digits=3):
	QE = som.quantization_error(X)
	TE = som.topographic_error(X)
	QE_ROUNDED = np.round(QE, digits)
	TE_ROUNDED = np.round(TE, digits)
	return QE, TE, QE_ROUNDED, TE_ROUNDED

def place_node_edges_via_weights(node_weights, alpha=.3, linewidth=1.5, ax=None):
	if ax is None:
		ax = plt.gca()

	rows, cols = node_weights.shape[:2]
	feat_x, feat_y = 0, 1

	for i in range(rows):
		for j in range(cols):
			current = node_weights[i, j]
			# vertical neighbor
			if i + 1 < rows:
				neighbor = node_weights[i+1, j]
				ax.plot([current[feat_x], neighbor[feat_x]],
						[current[feat_y], neighbor[feat_y]], 'k-', alpha=alpha, linewidth=linewidth)
			# horizontal neighbor
			if j + 1 < cols:
				neighbor = node_weights[i, j+1]
				ax.plot([current[feat_x], neighbor[feat_x]],
						[current[feat_y], neighbor[feat_y]], 'k-', alpha=alpha, linewidth=linewidth)

def place_node_edges(som, alpha=.3, linewidth=1.5, ax=None):
	if ax is None:
		ax = plt.gca()

	node_weights = som.get_weights()
	rows, cols = node_weights.shape[:2]
	feat_x, feat_y = 0, 1

	for i in range(rows):
		for j in range(cols):
			current = node_weights[i, j]
			# vertical neighbor
			if i + 1 < rows:
				neighbor = node_weights[i+1, j]
				ax.plot([current[feat_x], neighbor[feat_x]],
						[current[feat_y], neighbor[feat_y]], 'k-', alpha=alpha, linewidth=linewidth)
			# horizontal neighbor
			if j + 1 < cols:
				neighbor = node_weights[i, j+1]
				ax.plot([current[feat_x], neighbor[feat_x]],
						[current[feat_y], neighbor[feat_y]], 'k-', alpha=alpha, linewidth=linewidth)

def plot_som_convergence_over_epochs(X, epoch_min=1, epoch_max=10, step=1, te_ceiling=0.1, figsize=(10, 3), verb=False, show_fig=True, **hyparams):
	from _utils.som_embedding import SOM_Embedding

	hyparams = hyparams.copy()
	epoch_axis = np.arange(epoch_min, (epoch_max + 1), step)
	MQEs = []
	TEs = []

	for ni in epoch_axis:
		if verb: print(f"Training SOM for {ni} epochs...")

		hyparams["num_iteration"] = ni
		som = SOM_Embedding(**hyparams, verb=False).fit(X).som_
		QE = som.quantization_error(X)
		TE = som.topographic_error(X)

		MQEs.append(QE)
		TEs.append(TE)
	
	fig, ax1 = plt.subplots(figsize=figsize)
	ax2 = ax1.twinx()

	ax = ax1
	ax.scatter(epoch_axis, MQEs, color="cornflowerblue")
	ax.plot(epoch_axis, MQEs, color="cornflowerblue", label="QE")
	ax.set_xticks(epoch_axis)

	ax = ax2
	ax.scatter(epoch_axis, TEs, color="coral")
	ax.plot(epoch_axis, TEs, color="coral", label="TE")
	ax.hlines(te_ceiling, xmin=epoch_min, xmax=epoch_max, linestyle="dashed", color="grey", label="TE acceptance line")

	if show_fig:
		plt.legend()
		plt.show()

	return MQEs, TEs
