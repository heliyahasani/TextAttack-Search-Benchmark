import os
import datetime
from run_experiment import run
import argparse

MODELS = ["bert-base-uncased-yelp", "bert-base-uncased-mr", "lstm-mr", "lstm-yelp"]
MODEL_RESULT = {
    "bert-base-uncased-mr": "bert-mr-test",
    "bert-base-uncased-yelp": "bert-yelp-test",
    "lstm-yelp": "lstm-yelp-test",
    "lstm-mr": "lstm-mr-test",
}
TRANSFORMATIONS = ["word-swap-wordnet", "word-swap-embedding", "word-swap-hownet"]
CONSTRAINT_LEVEL = ["strict"]
SEARCH_METHODS = {
    "tabu_classic": ["tabu4", "tabu8", "tabu16"],
    "tabu_agglomerative": [
        "tabu_agglomerative_average",
        "tabu_agglomerative_complete",
        "tabu_agglomerative_single",
        "tabu_agglomerative_ward",
    ],
    "tabu_dbscan": ["tabu_dbscan"],
    "tabu_dynamic_tenure": ["tabu_dynamic_tenure", "tabu_dynamic"],
    "tabu_hdbscan": ["tabu_hdbscan"],
    "tabu_kmeans": ["tabu_kmeans"],
    "tabu_semantic_similarity": ["tabu_semantic_similarity"],
}


print(f"Running experiment for models {MODELS}")


def main(output_dir):
    for model in MODELS:
        for transformation in TRANSFORMATIONS:
            for constraint in CONSTRAINT_LEVEL:
                for family in SEARCH_METHODS:
                    for search in SEARCH_METHODS[family]:
                        recipe_path = f"recipes/{transformation}/{constraint}/{family}/{search}-recipe.py"
                        result_file_name = (
                            f"greedyWIR_{search}"
                            if family == "greedy-word-wir"
                            else search
                        )  # Define the log paths
                        exp_base_name = (
                            f"{MODEL_RESULT[model]}/{transformation}/{constraint}"
                        )
                        os.makedirs(exp_base_name, exist_ok=True)
                        txt_log_path = os.path.join(exp_base_name, "log.txt")
                        csv_log_path = os.path.join(exp_base_name, "log.csv")

                        start_time = datetime.datetime.now()
                        print(
                            f"Starting: Model={model}, Transformation={transformation}, Method={search}"
                        )

                        # Call the run function from the imported module
                        run(model, recipe_path, txt_log_path, csv_log_path)

                        end_time = datetime.datetime.now()
                        duration = end_time - start_time
                        with open(txt_log_path, "a") as f:
                            f.write(f"Experiment completed in {duration}.\n")
                        print(
                            f"Completed: Model={model}, Transformation={transformation}, Method={search}"
                        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all combinations of experiments.")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to store all outputs"
    )
    args = parser.parse_args()
    main(args.output_dir)
