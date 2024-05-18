from textattack.search_methods import SearchMethod
import numpy as np
from textattack.goal_function_results import GoalFunctionResultStatus
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN


class EmbeddingBasedTabuSearch(SearchMethod):
    def __init__(self, eps=0.5, min_samples=5):
        super().__init__()
        self.vectorizer = TfidfVectorizer()
        self.cluster_model = DBSCAN(eps=eps, min_samples=min_samples)
        self.tabu_clusters = set()
        self.is_vectorizer_fitted = False
        self.eps = eps

    def fit_vectorizer(self, corpus):
        corpus = [
            doc.printable_text() if hasattr(doc, "printable_text") else doc
            for doc in corpus
        ]
        self.vectorizer.fit(corpus)
        self.is_vectorizer_fitted = True

    def extract_features(self, texts):
        texts = [
            text.printable_text() if hasattr(text, "printable_text") else text
            for text in texts
        ]
        if not self.is_vectorizer_fitted:
            self.fit_vectorizer(texts)
        return self.vectorizer.transform(texts).toarray()

    def perform_clustering(self, features):
        if (
            features.shape[0] < 2
        ):  # DBSCAN requires a minimum of 2 points to perform clustering
            return np.array(
                [-1] * features.shape[0]
            )  # Return all as noise if not enough samples
        return self.cluster_model.fit_predict(features)

    def perform_search(self, initial_result):
        if not self.is_vectorizer_fitted:
            self.fit_vectorizer([initial_result.attacked_text.printable_text()])

        beam = [initial_result.attacked_text]
        best_result = initial_result

        while not best_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
            potential_next_beam = []
            texts = [
                t.printable_text()
                for t in beam
                for t in self.get_transformations(
                    t, original_text=initial_result.attacked_text
                )
            ]
            if not texts:
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
                if (
                    scores[i] < self.eps and cluster_label != -1
                ):  # Do not tabu noise points
                    self.tabu_clusters.add(cluster_label)

        return best_result

    @property
    def is_black_box(self):
        return True
