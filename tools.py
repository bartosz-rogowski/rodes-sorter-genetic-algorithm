from datetime import datetime
import pandas as pd
import numpy as np


def load_input_file(filepath: str = "prety.txt") -> pd.DataFrame:
    column_names = ["nr", "length"]
    return pd.read_csv(filepath, sep=' ', names=column_names, index_col="nr")


def generate_input_file(filepath: str, n: int):
    with open(filepath, "w+") as file:
        for i in range(n):
            file.write(f"{i + 1} {round(50 * np.random.rand(), 2)}\n")


def save_solution_to_file(solution: np.ndarray, filepath=None) -> None:
    if filepath is None:
        filepath = "Rogowski_output_" \
                   + datetime.now().__str__().replace(" ", "_").replace(".", "_").replace(":", "_") \
                   + ".txt"
    np.savetxt(filepath, solution, fmt="%i", delimiter="\n")


def can_build_triangle(a: float, b: float, c: float) -> bool:
    if a < b + c and b < c + a and c < a + b:
        return True
    return False


def calculate_triangle_area(a: float, b: float, c: float) -> float:
    p: float = 0.5*(a+b+c)
    return np.sqrt(p*(p-a)*(p-b)*(p-c))


def get_stats_of_solution(solution, rodes_lengths):
    n = len(rodes_lengths)
    triples = [(solution[3 * i], solution[3 * i + 1], solution[3 * i + 2])
               for i in range(n // 3)]
    triangles: int = 0
    areas = -np.ones((n,))
    for triple in triples:
        a = rodes_lengths[triple[0] - 1]
        b = rodes_lengths[triple[1] - 1]
        c = rodes_lengths[triple[2] - 1]
        if can_build_triangle(a, b, c):
            areas[triangles] = calculate_triangle_area(a, b, c)
            triangles += 1
    areas = areas[areas > 0]
    st_dev = areas.std()
    print(f"{triangles = }, {st_dev = :.3f}, {areas.mean() = :.3f}")
    st_dev = 100_000 if np.isnan(st_dev) or np.isclose(st_dev, 0.) else st_dev
    fitness = (3 * triangles / n + np.tanh(1. / st_dev)) / 2.

    return triangles, st_dev, fitness


def partially_matched_crossover(parents, offspring_size, ga_instance):
    offspring_list = []
    idx = 0
    while idx != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        locus1, locus2 = np.sort(
            np.random.choice(np.arange(len(parent1)+1), size=2, replace=False)
        )

        def generate_offspring(parent_1, parent_2):
            offspring = np.zeros(
                shape=(len(parent_1),),
                dtype=type(parent_1[0])
            )

            offspring[locus1:locus2] = parent_2[locus1:locus2]

            outer_locus_list = np.concatenate([
                np.arange(0, locus1),
                np.arange(locus2, len(parent_1))
            ])

            for i in outer_locus_list:
                candidate = parent_1[i]
                while candidate in parent_2[locus1:locus2]:
                    candidate = parent_1[np.where(parent_2 == candidate)[0][0]]
                offspring[i] = candidate
            # print(f"{parent_1 = }\t{parent_2 = }\t{offspring = }")
            return offspring

        offspring_list.append(generate_offspring(parent1, parent2))
        # offspring_list.append(generate_offspring(parent2, parent1))
        idx += 1

    return np.array(offspring_list)
