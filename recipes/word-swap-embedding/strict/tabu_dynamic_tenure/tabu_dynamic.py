from textattack.search_methods import SearchMethod
from textattack.transformations import WordSwapWordNet
from scipy.spatial.distance import cosine
import numpy as np
from textattack.goal_function_results import GoalFunctionResultStatus
from sentence_transformers import SentenceTransformer
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO)


class DynamicTabuList:
    def __init__(
        self, model, base_size=10, max_size=20, base_threshold=0.6, max_threshold=0.9
    ):
        self.model = model
        self.base_size = base_size
        self.size = base_size
        self.max_size = max_size
        self.base_threshold = base_threshold
        self.threshold = base_threshold
        self.max_threshold = max_threshold
        self.tabu_items = []
        self.failure_streak = 0

    def ensure_1d(self, vector):
        """Ensure the vector is 1-D"""
        if vector.ndim != 1:
            return vector.flatten()
        return vector

    def add(self, item):
        item_embedding = self.model.encode(item)
        item_embedding = self.ensure_1d(item_embedding)
        if not any(self.is_semantically_similar(item, ti[1]) for ti in self.tabu_items):
            self.tabu_items.append((item, item_embedding))
            if len(self.tabu_items) > self.size:
                self.tabu_items.pop(0)
            return True
        return False

    def is_semantically_similar(self, text, tabu_embedding):
        text_embedding = self.model.encode(text)
        text_embedding = self.ensure_1d(text_embedding)
        tabu_embedding = self.ensure_1d(tabu_embedding)
        return 1 - cosine(text_embedding, tabu_embedding) > self.threshold

    def update_failure_streak(self, successful):
        if successful:
            self.failure_streak = 0
        else:
            self.failure_streak += 1
        self.adjust_size_and_threshold()

    def adjust_size_and_threshold(self):
        if self.failure_streak > 3:
            self.size = min(self.size + 1, self.max_size)
            self.threshold = min(
                self.threshold + 0.05, self.max_threshold
            )  # Increasing the threshold
        else:
            self.size = max(self.size - 1, self.base_size)
            self.threshold = max(
                self.threshold - 0.05, self.base_threshold
            )  # Decreasing the threshold


class EmbeddingBasedTabuSearch(SearchMethod):
    def __init__(
        self,
        model="all-MiniLM-L6-v2",
        base_tabu_size=10,
        max_tabu_size=20,
        base_threshold=0.5,
        max_threshold=0.9,
    ):
        self.model = SentenceTransformer(model)
        self.tabu_list = DynamicTabuList(
            self.model, base_tabu_size, max_tabu_size, base_threshold, max_threshold
        )

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
                    # Only add non-tabu transformations to the next beam
                    if self.tabu_list.add(t.text):
                        potential_next_beam.append(t)

            if not potential_next_beam:
                return best_result

            results, search_over = self.get_goal_results(potential_next_beam)
            scores = np.array([result.score for result in results])
            best_result_index = scores.argmax()
            best_result = results[best_result_index]

            # Update the tabu list failure streak based on the success of the current best result
            self.tabu_list.update_failure_streak(
                best_result.score > initial_result.score
            )

            if search_over:
                return best_result

            # Continue with the best few results based on scores
            beam = [potential_next_beam[i] for i in np.argsort(-scores)[:5]]

        return best_result

    @property
    def is_black_box(self):
        return True
