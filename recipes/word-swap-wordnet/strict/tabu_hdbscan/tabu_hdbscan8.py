from textattack.search_methods import SearchMethod
import numpy as np
from textattack.goal_function_results import GoalFunctionResultStatus
from sklearn.feature_extraction.text import TfidfVectorizer
import hdbscan


class EmbeddingBasedTabuSearch(SearchMethod):
    def __init__(self, min_cluster_size=8, min_samples=None):
        super().__init__()
        self.vectorizer = TfidfVectorizer()
        self.cluster_model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples
        )
        self.tabu_clusters = set()
        self.is_vectorizer_fitted = False

    def fit_vectorizer(self, corpus):
        # Ensure corpus is a list of strings
        corpus = [
            doc.printable_text() if hasattr(doc, "printable_text") else doc
            for doc in corpus
        ]
        self.vectorizer.fit(corpus)
        self.is_vectorizer_fitted = True

    def extract_features(self, texts):
        # Ensure texts are strings
        texts = [
            text.printable_text() if hasattr(text, "printable_text") else text
            for text in texts
        ]
        if not self.is_vectorizer_fitted:
            # Fit the vectorizer if not already fitted
            self.fit_vectorizer(texts)
        return self.vectorizer.transform(texts).toarray()

    def perform_clustering(self, features):
        if features.shape[0] < self.cluster_model.min_cluster_size:
            # Not enough samples for clustering, return all as noise
            return np.array([-1] * features.shape[0])
        return self.cluster_model.fit_predict(features)

    def perform_search(self, initial_result):
        if not self.is_vectorizer_fitted and hasattr(
            initial_result.attacked_text, "printable_text"
        ):
            self.fit_vectorizer([initial_result.attacked_text.printable_text()])

        beam = [initial_result.attacked_text]
        best_result = initial_result

        while not best_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
            potential_next_beam = []
            texts = []
            for text in beam:
                transformations = self.get_transformations(
                    text, original_text=initial_result.attacked_text
                )
                texts.extend([t.printable_text() for t in transformations])
                potential_next_beam.extend(transformations)

            if not potential_next_beam:
                return best_result

            features = self.extract_features(texts)
            cluster_labels = self.perform_clustering(features)

            filtered_transformations = [
                t
                for t, label in zip(potential_next_beam, cluster_labels)
                if label not in self.tabu_clusters or label == -1
            ]

            if not filtered_transformations:
                return best_result

            results, search_over = self.get_goal_results(filtered_transformations)
            scores = np.array([r.score for r in results])
            if scores.size == 0:
                return best_result

            best_result_index = scores.argmax()
            best_result = results[best_result_index]
            if search_over:
                return best_result

            best_indices = (-scores).argsort()[: len(beam)]
            beam = [filtered_transformations[i] for i in best_indices]

            for i in best_indices:
                cluster_label = cluster_labels[i]
                if scores[i] < 0.5 and cluster_label != -1:
                    self.tabu_clusters.add(cluster_label)

        return best_result

    @property
    def is_black_box(self):
        return True
