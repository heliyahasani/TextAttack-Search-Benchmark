from textattack.search_methods import SearchMethod
from textattack.transformations import WordSwapWordNet
from scipy.spatial.distance import cosine
import numpy as np
import os, sys
from nltk.tokenize import word_tokenize, sent_tokenize
from textattack.goal_function_results import GoalFunctionResultStatus
from recipes.word_swap_wordnet.strict.tabu_search.tabu import EmbeddingBasedTabuSearch

current_dir = os.path.dirname(os.path.realpath(__file__))
transformation_dir = os.path.normpath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(transformation_dir)
from transformation import TRANSFORMATION


import random
import math


class HybridGeneticTabuSearch(SearchMethod):
    def __init__(
        self, tabu_list_size=10, tabu_tenure=5, population_size=10, mutation_rate=0.1
    ):
        self.tabu_list_size = tabu_list_size
        self.tabu_tenure = tabu_tenure
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        super().__init__()

    def perform_search(self, initial_result):
        """Perform a hybrid genetic algorithm combined with tabu search."""
        best_result = initial_result
        tabu_list = []

        # Initialize population
        population = [initial_result.attacked_text] * self.population_size

        while not best_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
            # Apply tabu search on the current population
            population = self.tabu_search(population, initial_result, tabu_list)

            # Apply genetic operations (crossover and mutation)
            population = self.genetic_operations(population, initial_result)

            # Select the best individual as the result
            results, _ = self.get_goal_results(population)
            best_result = max(results, key=lambda x: x.score)

        return best_result

    def tabu_search(self, population, initial_result, tabu_list):
        """Apply tabu search on the population."""
        new_population = []
        for text in population:
            beam = [text]
            while not initial_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                # Perform tabu search on the current individual
                potential_next_beam = self.get_transformations(
                    beam[0], original_text=initial_result.attacked_text
                )
                potential_next_beam = [
                    t for t in potential_next_beam if t not in tabu_list
                ]

                if len(potential_next_beam) == 0:
                    break

                results, _ = self.get_goal_results(potential_next_beam)
                best_result = max(results, key=lambda x: x.score)
                beam = [best_result.attacked_text]

                # Update tabu list
                tabu_list.append(beam[0])
                if len(tabu_list) > self.tabu_list_size:
                    tabu_list.pop(0)

            new_population.append(beam[0])
        return new_population

    def genetic_operations(self, population, initial_result):
        """Apply genetic operations (crossover and mutation) on the population."""
        # Perform crossover
        offspring = []
        for _ in range(self.population_size // 2):
            parent1, parent2 = random.sample(population, 2)
            crossover_point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            offspring.extend([child1, child2])

        # Perform mutation
        mutated_offspring = []
        for individual in offspring:
            if random.random() < self.mutation_rate:
                mutation_point = random.randint(0, len(individual) - 1)
                mutated_individual = self.mutate(individual, mutation_point)
                mutated_offspring.append(mutated_individual)
            else:
                mutated_offspring.append(individual)

        return mutated_offspring

    def mutate(self, individual, mutation_point):
        """Mutate the individual."""
        new_individual = individual[:mutation_point]
        new_individual += random.choice(
            self.get_transformations(
                individual[mutation_point], original_text=individual
            )
        )[0]
        new_individual += individual[mutation_point + 1 :]
        return new_individual

    @property
    def is_black_box(self):
        return True
