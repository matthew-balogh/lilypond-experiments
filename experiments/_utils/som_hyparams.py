import numpy as np

SOM_TOPOLOGY = "rectangular"
SOM_USE_EPOCHS = True
SOM_RANDOM_ORDER = True
SOM_LEARNING_DECAY_FN = "linear_decay_to_zero"
SOM_SIGMA_DECAY_FN = "asymptotic_decay"
SOM_NUM_ITERATION = 20
SOM_LEARNING_RATE = .6

def calc_recommended_grid_size_in_total(X):
	N = len(X)
	dd = int(np.ceil(5 * np.sqrt(N)))
	return dd

def calc_recommended_side_length_ratio(X):
	cov = np.cov(X, rowvar=False)
	eigvals = np.linalg.eigvalsh(cov)
	eigvals = np.sort(eigvals)[::-1]
	# Take the two largest eigenvalues
	lambda1, lambda2 = eigvals[:2]
	ratio = np.sqrt(lambda1 / lambda2)
	return ratio

def calc_recommended_grid_size(X):
	M = calc_recommended_grid_size_in_total(X)
	ratio = calc_recommended_side_length_ratio(X)
	height = int(np.ceil(np.sqrt(M / ratio)))
	width = int(np.ceil(ratio * height))
	return height, width

def calc_initial_sigma(d1, d2, factor=3.0):
    L = max(d1, d2)
    if L <= 1: return 1.0
    return np.round(L / factor, 2)

def obtain_som_hyparams(X, verb=True):
    total = calc_recommended_grid_size_in_total(X)
    if verb: print(f"SOM grid recommended total node count, based on {len(X)} training data points: {total}")

    height, width = calc_recommended_grid_size(X)
    if verb: print(f"SOM grid recommended dimensions: width={width}, height={height}")

    som_dim_1 = height
    som_dim_2 = width

    som_sigma = calc_initial_sigma(som_dim_1, som_dim_2)

    hyperparams = {
        "d1": som_dim_1,
        "d2": som_dim_2,
        "sigma": som_sigma,
        "learning_rate": SOM_LEARNING_RATE,
        "num_iteration": SOM_NUM_ITERATION,
        "decay_function": SOM_LEARNING_DECAY_FN,
        "sigma_decay_function": SOM_SIGMA_DECAY_FN,
    }

    if verb:
        print("SOM hyperparameters determined as (d1, d2, sigma, learning rate, num. iteration):",
              som_dim_1, som_dim_2, som_sigma, SOM_LEARNING_RATE, SOM_NUM_ITERATION)

    return hyperparams
