import os
import time
import run_experiment

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
    "beam-search": ["greedy", "beam4", "beam8", "beam16", "beam32", "beam64"],
    "greedy-word-wir": ["delete", "unk", "pwws", "gradient", "random"],
    "population": ["genetic", "pso"],
}

print(f"Running experiment for models: {MODELS}")

for model in MODELS:
    for transformation in TRANSFORMATIONS:
        for constraint in CONSTRAINT_LEVEL:
            for family in SEARCH_METHODS:
                for search in SEARCH_METHODS[family]:
                    recipe_path = f"recipes/{transformation}/{constraint}/{family}/{search}-recipe.py"
                    result_file_name = (
                        f"greedyWIR_{search}" if family == "greedy-word-wir" else search
                    )
                    exp_base_name = (
                        f"{MODEL_RESULT[model]}/{transformation}/{constraint}"
                    )
                    result_dir = f"results/{exp_base_name}"
                    chkpt_dir = f"end-checkpoints/{exp_base_name}"
                    if not os.path.exists(result_dir):
                        os.makedirs(result_dir)
                    if not os.path.exists(chkpt_dir):
                        os.makedirs(chkpt_dir)

                    log_txt_path = f"{result_dir}/{result_file_name}.txt"
                    log_csv_path = f"{result_dir}/{result_file_name}.csv"
                    chkpt_path = f"{exp_base_name}/{result_file_name}"

                    # Measure the start time
                    start_time = time.time()

                    run_experiment.run(
                        model,
                        recipe_path,
                        log_txt_path,
                        log_csv_path,
                        chkpt_path=chkpt_path,
                    )

                    # Measure the end time
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    elapsed_minutes = elapsed_time / 60

                    # Print and log the elapsed time
                    print(
                        f"Experiment {model} {transformation} {constraint} {search} completed in {elapsed_minutes:.2f} minutes."
                    )

                    # Append the time taken to a log file
                    with open(f"{result_dir}/timing_log.txt", "a") as log_file:
                        log_file.write(
                            f"{model}, {transformation}, {constraint}, {search}: {elapsed_minutes:.2f} minutes\n"
                        )
