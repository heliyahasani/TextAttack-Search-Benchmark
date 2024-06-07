from textattack.search_methods import SearchMethod
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textattack.goal_function_results import GoalFunctionResultStatus


class EmbeddingBasedTabuSearch(SearchMethod):
    def __init__(self, num_clusters=8):
        super().__init__()
        self.num_clusters = num_clusters
        self.vectorizer = TfidfVectorizer()
        self.cluster_model = KMeans(n_clusters=self.num_clusters)
        self.tabu_clusters = set()
        self.is_vectorizer_fitted = False

    def fit_vectorizer(self, corpus):
        self.vectorizer.fit(corpus)
        self.is_vectorizer_fitted = True

    def extract_features(self, texts):
        return self.vectorizer.transform(texts).toarray()

    def perform_clustering(self, features):
        unique_features = np.unique(features, axis=0)
        n_clusters = min(len(unique_features), self.num_clusters)
        if n_clusters < self.num_clusters:
            self.cluster_model = KMeans(n_clusters=n_clusters)
        cluster_labels = self.cluster_model.fit_predict(unique_features)
        return cluster_labels

    def perform_search(self, initial_result):
        if not self.is_vectorizer_fitted:
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
                if label not in self.tabu_clusters
            ]

            if not filtered_transformations:
                return best_result  # No valid transformations to process

            results, search_over = self.get_goal_results(filtered_transformations)
            scores = np.array([r.score for r in results])
            if scores.size == 0:
                return best_result  # Handle empty scores sequence

            best_result_index = scores.argmax()
            best_result = results[best_result_index]
            if search_over:
                return best_result

            best_indices = (-scores).argsort()[: len(beam)]
            beam = [filtered_transformations[i] for i in best_indices]

            for i in best_indices:
                cluster_label = cluster_labels[i]
                self.tabu_clusters.add(cluster_label)

        return best_result

    @property
    def is_black_box(self):
        return True
