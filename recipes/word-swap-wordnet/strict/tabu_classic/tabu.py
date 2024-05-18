from textattack.search_methods import SearchMethod
import numpy as np
import os, sys
from textattack.goal_function_results import GoalFunctionResultStatus

current_dir = os.path.dirname(os.path.realpath(__file__))
transformation_dir = os.path.normpath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(transformation_dir)


class EmbeddingBasedTabuSearch(SearchMethod):
    def perform_search(self, initial_result, tabu_list_size, tabu_tenur):
        """Perform a focused tabu search on synonyms of words in the sentence, with tabu list and aspiration criteria."""
        beam = [initial_result.attacked_text]
        best_result = initial_result
        tabu_list = []
        tabu_list_size = tabu_list_size  # Example size, adjust based on your needs
        tabu_tenur = tabu_tenur

        while not best_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
            potential_next_beam = []
            for text in beam:
                transformations = self.get_transformations(
                    text, original_text=initial_result.attacked_text
                )
                # Exclude transformations in the tabu list
                transformations = [t for t in transformations if t not in tabu_list]
                potential_next_beam += transformations
            # print("potential", potential_next_beam)

            if len(potential_next_beam) == 0:
                return best_result

            results, search_over = self.get_goal_results(potential_next_beam)
            scores = np.array([r.score for r in results])
            best_result = results[scores.argmax()]
            if search_over:
                return best_result

            best_indices = (-scores).argsort()[:tabu_tenur]
            beam = [potential_next_beam[i] for i in best_indices]

            # Update tabu list with the selected transformations
            for i in best_indices:
                if potential_next_beam[i] not in tabu_list:
                    if len(tabu_list) >= tabu_list_size:
                        tabu_list.pop(0)
                    tabu_list.append(potential_next_beam[i])

        return best_result

    @property
    def is_black_box(self):
        return True


# python run_experiment.py --model bert-base-uncased-mr --recipe-path /home/heliya/Desktop/thesis/thesis/TextAttack-Search-Benchmark/recipes/word_swap_wordnet/strict/tabu_search/tabu_recipe_geometric_annealing.py --txt-log-path . --csv-log-path .
