from dataclasses import dataclass

import logging
import math

from scipy.stats import qmc
import numpy as np
import pandas as pd

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


@dataclass
class Population:
    population_size: int
    population_parameters: dict
    seed: int = None

    def __post_init__(self):
        self.dim = len(self.population_parameters)
        self._seen = set()

        if self.seed is not None:
            np.random.seed(self.seed)

    # def clear_duplicates(self, population: pd.DataFrame) -> pd.DataFrame:
    #     unique = []
    #     for ind in population:

    def create_population(self, pop_size: int = None) -> pd.DataFrame:

        if pop_size is None:
            pop_size = self.population_size

        n_power2 = 2 ** math.ceil(math.log2(pop_size))

        sobol = qmc.Sobol(d=self.dim, scramble=True, rng=self.seed)
        sobol_samples = sobol.random(n=n_power2, workers=-1)[:pop_size]

        lhs = qmc.LatinHypercube(d=self.dim, rng=self.seed)
        lhs_samples = lhs.random(pop_size, workers=-1)

        for i in range(self.dim):
            order = np.argsort(sobol_samples[:, i])
            lhs_samples[:, i] = np.sort(lhs_samples[:, i])[order]

        l_bounds = np.array([v[0] for v in self.population_parameters.values()])
        u_bounds = np.array([v[1] for v in self.population_parameters.values()])
        scaled_samples = qmc.scale(lhs_samples, l_bounds, u_bounds)

        df = pd.DataFrame(
            scaled_samples, columns=list(self.population_parameters.keys())
        )

        df = df.round(3)

        return df


if __name__ == "__main__":
    test_params = {
        "vehicle.shocks.lf.base_model.compression_clicks_high_speed": {
            "limits": [0, 40],
            "type": "int",
        },
        "vehicle.setup.setup_height_lf": {"limits": [2.1, 2.4], "type": "float"},
        "vehicle.setup.setup_height_lr": {"limits": [1.9, 2.2], "type": "float"},
        "vehicle.setup.setup_height_rf": {"limits": [3.6, 3.9], "type": "float"},
        "vehicle.setup.setup_height_rr": {"limits": [3.4, 3.7], "type": "float"},
        "vehicle.springs.linear_rate_override_lf": {
            "limits": [600.0, 1800.0],
            "type": "float",
        },
        "vehicle.springs.linear_rate_override_lr": {
            "limits": [500.0, 1800.0],
            "type": "float",
        },
        "vehicle.springs.linear_rate_override_rf": {
            "limits": [600.0, 1500.0],
            "type": "float",
        },
        "vehicle.springs.linear_rate_override_rr": {
            "limits": [500.0, 800.0],
            "type": "float",
        },
        "vehicle.setup.nose_weight": {"limits": [49.1, 49.8], "type": "float"},
        "vehicle.setup.cross_weight": {"limits": [48.0, 62.0], "type": "float"},
        "vehicle.setup.post_setup_arblink_length_adj_lf": {
            "limits": [-0.5, 0.5],
            "type": "float",
        },
        "vehicle.setup.post_setup_arblink_length_adj_rr": {
            "limits": [-1.0, 0.5],
            "type": "float",
        },
        "vehicle.antirollbars.front_stiffness": {
            "limits": [0.0, 1115.0],
            "type": "float",
        },
        "vehicle.antirollbars.rear_stiffness": {
            "limits": [0.0, 1090.0],
            "type": "float",
        },
    }

    generator = Population(
        population_size=20,
        population_parameters=test_params,
        seed=84,
    )
    pop = generator.create_population()
    pop = pop.iloc[:10]
    print(pop)
    pop.iloc[-2] = pop.iloc[-1]
    pop.iloc[-5] = pop.iloc[-3]
    print(pop)

    # print(pop)

    generator.repair_population(pop)

    # new_pop = generator.evolve_population(pop[0:13])
    # # new_pop = generator.evolve_population(pop)

    # print(new_pop)