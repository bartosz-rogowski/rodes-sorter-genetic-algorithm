import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from typing import Callable
from tools import load_input_file, can_build_triangle, \
    calculate_triangle_area, get_stats_of_solution, partially_matched_crossover
import pygad
import warnings


class GeneticAlgorithm:
    def __init__(self, rodes_lengths, **kwargs):
        self.rodes_lengths = rodes_lengths
        self.n = len(self.rodes_lengths)
        gene_space = [i for i in range(1, n + 1)]

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
                       for i in range(n // 3)]
            triangles: int = 0
            areas = -np.ones((n,))
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
            fitness = (3*triangles/n + 0.75*np.tanh(20. / st_dev)) / 2.
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


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    rodes_df: pd.DataFrame = load_input_file()
    n: int = len(rodes_df)
    assert n >= 3, "Liczba prętów musi być większa niż 2"
    assert n % 3 == 0, "Liczba prętów musi być podzielna 3"
    rodes_lengths: np.ndarray = rodes_df["length"].to_numpy()
    genetic_algorithm = GeneticAlgorithm(
        rodes_lengths=rodes_lengths,
        num_generations=100,
        num_parents_mating=3,
        sol_per_pop=100,
        num_genes=n,
        gene_type=np.int32,
        mutation_probability=5e-2,
        parent_selection_type="tournament",
        crossover_type=partially_matched_crossover,
        mutation_type="swap",
        allow_duplicate_genes=False,
    )

    repeats: int = 10
    triangles_list = np.zeros(shape=(repeats,), dtype="int")
    st_dev_list = np.zeros(shape=(repeats,),)
    for i in range(repeats):
        genetic_algorithm.run()
        best_solution, best_solution_fitness, _ = genetic_algorithm.get_best_solution()
        print(f"{best_solution.tolist() = }\n{best_solution_fitness = }")
        triangles, std_dev, _ = get_stats_of_solution(best_solution, rodes_lengths)
        triangles_list[i] = triangles
        st_dev_list[i] = std_dev
    # genetic_algorithm.plot_fitness_function()
    plt.figure(figsize=(8, 8))
    plt.scatter(triangles_list, st_dev_list, marker='x')
    plt.title("najlepsze rozwiązania z 10 uruchomień")
    plt.xlabel("liczba utworzonych trójkątów")
    plt.xlim((-0.5, n//3+0.5))
    plt.ylabel("odchylenie")
    plt.grid()
    plt.show()
