import os
import datetime
from run_experiment import run

MODELS = ["bert-base-uncased-yelp", "bert-base-uncased-mr", "lstm-mr", "lstm-yelp"]
MODEL_RESULT = {
    "bert-base-uncased-mr": "bert-mr-test",
    "bert-base-uncased-yelp": "bert-yelp-test",
    "lstm-yelp": "lstm-yelp-test",
    "lstm-mr": "lstm-mr-test",
}
TRANSFORMATIONS = ["word-swap-wordnet", "word-swap-hownet"]  # "word-swap-embedding"
CONSTRAINT_LEVEL = ["strict"]
SEARCH_METHODS = {
    "tabu_classic": ["tabu4", "tabu8", "tabu16"],
}

print(f"Running experiment for models {MODELS}")

for model in MODELS:
    for transformation in TRANSFORMATIONS:
        for constraint in CONSTRAINT_LEVEL:
            for family in SEARCH_METHODS:
                for search in SEARCH_METHODS[family]:
                    recipe_path = f"recipes/{transformation}/{constraint}/{family}/{search}_recipe.py"
                    result_file_name = (
                        f"greedyWIR_{search}" if family == "greedy-word-wir" else search
                    )
                    exp_base_name = (
                        f"{MODEL_RESULT[model]}/{transformation}/{constraint}"
                    )
                    result_dir = f"results/{exp_base_name}"
                    chkpt_dir = f"end-checkpoints/{exp_base_name}"
                    os.makedirs(result_dir, exist_ok=True)
                    os.makedirs(chkpt_dir, exist_ok=True)

                    log_txt_path = f"{result_dir}/{result_file_name}.txt"
                    log_csv_path = f"{result_dir}/{result_file_name}.csv"
                    chkpt_path = f"{chkpt_dir}/{result_file_name}"

                    start_time = datetime.datetime.now()
                    run(
                        model,
                        recipe_path,
                        log_txt_path,
                        log_csv_path,
                        chkpt_path=chkpt_path,
                    )
                    end_time = datetime.datetime.now()

                    # Calculate the duration and append it to the log file
                    duration = end_time - start_time
                    with open(log_txt_path, "a") as f:
                        f.write(f"Experiment completed in {duration}.\n")
