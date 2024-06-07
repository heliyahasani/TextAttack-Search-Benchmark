import numpy as np
from textattack.search_methods import SearchMethod
from textattack.goal_function_results import GoalFunctionResultStatus


class EmbeddingBasedTabuSearch(SearchMethod):
    def perform_search(self, initial_result, tabu_list_size=4, tabu_tenure_size=4):
        beam = [initial_result.attacked_text]
        best_result = initial_result
        tabu_list = []
        tabu_list_size = tabu_list_size
        tabu_tenure = {text: tabu_tenure_size for text in beam}  # Initial tenure

        while not best_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
            potential_next_beam = []
            for text in beam:
                transformations = self.get_transformations(
                    text, original_text=initial_result.attacked_text
                )
                transformations = [t for t in transformations if t not in tabu_list]
                potential_next_beam += transformations

            if len(potential_next_beam) == 0:
                return best_result

            results, search_over = self.get_goal_results(potential_next_beam)
            scores = np.array([r.score for r in results])
            best_result = results[scores.argmax()]
            if search_over:
                return best_result

            # Update beam based on best scores
            best_indices = (-scores).argsort()[: len(beam)]
            beam = [potential_next_beam[i] for i in best_indices]

            # Update tabu list with the selected transformations and their performances
            for i in best_indices:
                trans = potential_next_beam[i]
                if trans not in tabu_list:
                    if len(tabu_list) >= tabu_list_size:
                        # Remove the oldest or least effective transformation
                        least_effective = min(tabu_tenure, key=tabu_tenure.get)
                        tabu_list.remove(least_effective)
                        tabu_tenure.pop(least_effective)
                    tabu_list.append(trans)
                    tabu_tenure[trans] = 5  # Reset tenure for new entries
                else:
                    # Adjust tenure based on performance, e.g., reduce if very effective
                    if scores[i] > np.mean(scores):
                        tabu_tenure[trans] = max(1, tabu_tenure[trans] - 1)

            # Decrement tenure and remove expired items from tabu list
            for t in list(
                tabu_tenure.keys()
            ):  # Use keys() to avoid modification during iteration
                tabu_tenure[t] -= 1
                if tabu_tenure[t] <= 0:
                    if t in tabu_list:  # Check before removing
                        tabu_list.remove(t)
                    tabu_tenure.pop(t)

        return best_result

    @property
    def is_black_box(self):
        return True


# In this code, we have implemented a mechanism where each transformation's tenure is dynamically adjusted based on its score relative to the average score.
# Transformations that perform well might have their tenure reduced to be reevaluated sooner. This helps in maintaining a more adaptive and
# responsive tabu list. Additionally, transformations are removed from the tabu list based on their tenure, allowing for more dynamic exploration of the search space.
