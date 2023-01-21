import warnings
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tools import load_input_file, get_stats_of_solution, \
    partially_matched_crossover, generate_input_file, save_solution_to_file
from genetic_algorithm import GeneticAlgorithm


def main(filepath: str = "prety.txt"):
    rodes_df: pd.DataFrame = load_input_file(filepath)
    n: int = len(rodes_df)
    assert n >= 3, "Liczba prętów musi być większa niż 2"
    assert n % 3 == 0, "Liczba prętów musi być podzielna 3"
    rodes_lengths: np.ndarray = rodes_df["length"].to_numpy()
    genetic_algorithm = GeneticAlgorithm(
        rodes_lengths=rodes_lengths,
        num_generations=200,
        num_parents_mating=50,
        sol_per_pop=100,
        num_genes=n,
        gene_type=np.int16,
        mutation_probability=5e-2,
        parent_selection_type="tournament",
        crossover_type=partially_matched_crossover,
        mutation_type="swap",
        allow_duplicate_genes=False,
    )

    genetic_algorithm.run()
    best_solution, best_solution_fitness, _ = genetic_algorithm.get_best_solution()
    print(f"{best_solution_fitness = }")
    triangles, std_dev, _ = get_stats_of_solution(best_solution, rodes_lengths)
    genetic_algorithm.plot_fitness_function()
    save_solution_to_file(best_solution)
    return triangles, std_dev


def run_main_n_times(repeats: int, plot_stats: bool = True):
    triangles_list = np.zeros(shape=(repeats,), dtype="int")
    st_dev_list = np.zeros(shape=(repeats,),)
    for i in range(repeats):
        triangles, std_dev = main()
        triangles_list[i] = triangles
        st_dev_list[i] = std_dev
        print()
    if plot_stats:
        plt.figure(figsize=(8, 8))
        plt.scatter(triangles_list, st_dev_list, marker='x')
        plt.title("najlepsze rozwiązania z 10 uruchomień")
        plt.xlabel("liczba utworzonych trójkątów")
        plt.xlim((-0.5, 300//3+0.5))
        plt.ylabel("odchylenie")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    start_time = time.perf_counter()
    main()
    # run_main_n_times(repeats=10, plot_stats=True)
    end_time = time.perf_counter()
    print(f"Program executed in {(end_time - start_time):.2f} seconds")
    plt.show()
