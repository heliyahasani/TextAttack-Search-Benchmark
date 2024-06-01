from textattack.search_methods import SearchMethod
from textattack.transformations import WordSwapWordNet
from scipy.spatial.distance import cosine
import numpy as np
from textattack.goal_function_results import GoalFunctionResultStatus
from sentence_transformers import SentenceTransformer


class EmbeddingBasedTabuSearch(SearchMethod):
    def __init__(self, model="all-MiniLM-L6-v2", tabu_size=4, threshold=0.5):

        self.model = SentenceTransformer(model)
        self.tabu_list = []
        self.tabu_size = tabu_size
        self.threshold = threshold

    def is_semantically_similar(self, text, tabu_item):
        text_embedding = self.model.encode(text)
        tabu_embedding = self.model.encode(tabu_item)
        return 1 - cosine(text_embedding, tabu_embedding) > self.threshold

    def perform_search(self, initial_result):
        beam = [initial_result.attacked_text]
        best_result = initial_result

        while not best_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
            potential_next_beam = []
            for text in beam:
                transformations = self.get_transformations(
                    text, original_text=initial_result.attacked_text
                )
            for t in transformations:
                if all(
                    not self.is_semantically_similar(t.text, ti)
                    for ti in self.tabu_list
                ):
                    potential_next_beam.append(t)

            if not potential_next_beam:
                return best_result

            results, search_over = self.get_goal_results(potential_next_beam)
            scores = np.array([r.score for r in results])
            best_result_index = scores.argmax()
            best_result = results[best_result_index]

            if search_over:
                return best_result

            if len(self.tabu_list) >= self.tabu_size:
                self.tabu_list.pop(0)
            self.tabu_list.append(potential_next_beam[best_result_index].text)

        return best_result

    @property
    def is_black_box(self):
        return True
