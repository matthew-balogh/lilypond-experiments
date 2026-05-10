import numpy as np

from _utils.som import train_som
from _utils.som_hyparams import SOM_RANDOM_ORDER, SOM_TOPOLOGY, SOM_USE_EPOCHS, SOM_LEARNING_RATE, SOM_NUM_ITERATION, SOM_LEARNING_DECAY_FN, SOM_SIGMA_DECAY_FN
from _utils.constants import RANDOM_SEED

class SOM_Embedding():
    def __init__(self,
                    d1, d2, sigma, continuity_violation_limit=0.1,
                    topology=SOM_TOPOLOGY, learning_rate=SOM_LEARNING_RATE, num_iteration=SOM_NUM_ITERATION,
                    decay_function=SOM_LEARNING_DECAY_FN, sigma_decay_function=SOM_SIGMA_DECAY_FN,
                    use_epochs=SOM_USE_EPOCHS, random_order=SOM_RANDOM_ORDER, random_seed=RANDOM_SEED, verb=True):
        
        super().__init__()
        self.d1 = d1
        self.d2 = d2
        self.sigma = sigma
        self.continuity_violation_limit = continuity_violation_limit
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration
        self.topology = topology
        self.use_epochs = use_epochs
        self.decay_function = decay_function
        self.sigma_decay_function = sigma_decay_function
        self.random_order = random_order
        self.random_seed = random_seed
        self.verb = verb

    def fit(self, X, y=None):
        hyperparams = {
            "d1": self.d1,
            "d2": self.d2,
            "sigma": self.sigma,
            "topology": self.topology,
            "learning_rate": self.learning_rate,
            "num_iteration": self.num_iteration,
            "decay_function": self.decay_function,
            "sigma_decay_function": self.sigma_decay_function,
            "use_epochs": self.use_epochs,
            "random_order": self.random_order,
            "random_seed": self.random_seed,
            "verb": self.verb,
        }

        som = train_som(X, **hyperparams)

        if self.continuity_violation_limit is not None:
            assert som.topographic_error(X) < self.continuity_violation_limit, f"The SOM is not organized. Topographic error: {som.topographic_error(X)}, limit: {self.continuity_violation_limit}"

        self.som_ = som
        return self

    def quantization_errors(self, X):
        quantization_errors =  np.linalg.norm(X - self.som_.quantization(X), axis=1)
        return quantization_errors
