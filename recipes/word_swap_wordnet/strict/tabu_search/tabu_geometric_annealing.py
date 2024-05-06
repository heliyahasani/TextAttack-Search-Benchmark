from textattack.search_methods import SearchMethod
from textattack.goal_function_results import GoalFunctionResultStatus
import numpy as np
import random
import math


class HybridTabuSearch(SearchMethod):
    def __init__(
        self,
        tabu_list_size=10,
        tabu_tenure=5,
        initial_temperature=1.0,
        cooling_rate=0.9,
    ):
        """
        Initializes the Hybrid Tabu Search with Simulated Annealing parameters.
        :param tabu_list_size: Maximum size of the tabu list.
        :param tabu_tenure: Number of iterations a move remains in the tabu list.
        :param initial_temperature: Starting temperature for simulated annealing.
        :param cooling_rate: Rate at which the temperature decreases, must be between 0 and 1.
        """
        self.tabu_list_size = tabu_list_size
        self.tabu_tenure = tabu_tenure
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        super().__init__()

    def perform_search(self, initial_result):
        """
        Perform a hybrid tabu search combined with simulated annealing on the input text.
        :param initial_result: The initial text to be attacked.
        """
        current_result = initial_result
        best_result = initial_result
        tabu_list = []

        iteration = 0
        temperature = self.initial_temperature

        while not best_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
            transformations = self.get_transformations(
                current_result.attacked_text, original_text=initial_result.attacked_text
            )

            # Exclude transformations in the tabu list
            transformations = [t for t in transformations if t not in tabu_list]

            if not transformations:
                # If no valid transformations are available, exit the search
                break

            # Evaluate transformations
            results, search_over = self.get_goal_results(transformations)
            if search_over:
                return best_result

            # Selection based on scores and simulated annealing
            scores = np.array([r.score for r in results])
            if np.max(scores) > best_result.score:
                best_result_index = np.argmax(scores)
                best_result = results[best_result_index]
                current_result = results[best_result_index]

            # Simulated annealing to potentially accept worse solutions
            elif random.random() < math.exp(
                (best_result.score - np.max(scores)) / temperature
            ):
                random_choice_index = random.choice([i for i, _ in enumerate(scores)])
                current_result = results[random_choice_index]

            # Update the tabu list
            tabu_list.append(current_result.attacked_text)
            if len(tabu_list) > self.tabu_list_size:
                tabu_list.pop(0)

            # Cooling down the temperature
            temperature *= self.cooling_rate
            iteration += 1

        return best_result

    @property
    def is_black_box(self):
        return True
