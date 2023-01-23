from typing import Callable
import pygad
from tools import can_build_triangle, calculate_triangle_area
from matplotlib import pyplot as plt
import numpy as np


class GeneticAlgorithm:
    def __init__(self, rodes_lengths, **kwargs):
        self.rodes_lengths = rodes_lengths
        self.n = len(self.rodes_lengths)
        gene_space = [i for i in range(1, self.n + 1)]

        self.ga_instance = pygad.GA(
            fitness_func=self.__calculate_fitness_wrapper(),
            gene_space=gene_space,
            **kwargs
        )

    def run(self):
        self.ga_instance.run()

    def __calculate_fitness_wrapper(self) -> Callable[[np.ndarray, int], float]:
        """Private wrapper method for function calculating fitness.
        It is needed in order to use `self.rodes_lengths` parameter

        :return: function calculating fitness fulfilling pygad requirements
        """
        def calculate_fitness(solution: np.ndarray, solution_idx: int) -> float:
            triples = [(solution[3*i], solution[3*i + 1], solution[3*i + 2])
                       for i in range(self.n // 3)]
            triangles: int = 0
            areas = -np.ones((self.n,))
            for triple in triples:
                a = self.rodes_lengths[triple[0] - 1]
                b = self.rodes_lengths[triple[1] - 1]
                c = self.rodes_lengths[triple[2] - 1]
                if can_build_triangle(a, b, c):
                    areas[triangles] = calculate_triangle_area(a, b, c)
                    triangles += 1
            areas = areas[areas > 0]
            st_dev = areas.std()
            # print(solution)
            # print(f"{triangles = }, {st_dev = }, {areas.mean() = }")
            st_dev = 100_000 if np.isnan(st_dev) or np.isclose(st_dev, 0.) \
                else st_dev
            fitness = 0.5 * (3*triangles/self.n + np.tanh(10. / st_dev))
            return fitness
        return calculate_fitness

    def get_best_solution(self):
        return self.ga_instance.best_solution()

    def plot_fitness_function(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.ga_instance.best_solutions_fitness, marker='.')
        plt.xlabel("numer pokolenia")
        plt.ylabel("wartość funkcji przystosowania")
        plt.title("Wartość przystosowania w funkcji numeru pokolenia")
        # plt.ylim((-0.05, 1.05))
        plt.grid()
        return plt.gcf()

