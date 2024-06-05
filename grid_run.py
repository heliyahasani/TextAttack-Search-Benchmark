import datetime
import os
import subprocess
from time import sleep
import run_experiment

MODELS = ["bert-base-uncased-yelp", "bert-base-uncased-mr"]
# MODELS = ["lstm-mr", "lstm-yelp"]

MODEL_RESULT = {
    "bert-base-uncased-mr": "bert-mr-test",
    "bert-base-uncased-yelp": "bert-yelp-test",
    "lstm-yelp": "lstm-yelp-test",
    "lstm-mr": "lstm-mr-test",
}
# TRANSFORMATIONS = ["word-swap-wordnet", "word-swap-embedding", "word-swap-hownet"]
# TRANSFORMATIONS = ["word-swap-wordnet"]
TRANSFORMATIONS = ["word-swap-embedding"]
# TRANSFORMATIONS = ["word-swap-hownet"]
CONSTRAINT_LEVEL = ["strict"]
SEARCH_METHODS = {
    # "tabu_dynamic_tenure": ["tabu_dynamic_tenure]
    "tabu_semantic_similarity": ["tabu_semantic_similarity"],
}


print(f'Running experiment for model "{MODELS}"')


procs = []

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

                    start_time = datetime.datetime.now()
                    print(
                        f"Starting: Model={model}, Transformation={transformation}, Method={search}"
                    )

                    proc = run_experiment.run(
                        model,
                        recipe_path,
                        log_txt_path,
                        log_csv_path,
                        chkpt_path=chkpt_path,
                    )
                    # proc = subprocess.Popen(["./test.sh", exp_base_name])
                    procs.append(
                        (
                            proc,
                            {
                                "start_time": start_time,
                                "model": model,
                                "transformation": transformation,
                                "search_method": search,
                                "exp_base_name": exp_base_name,
                            },
                        )
                    )

for p in procs:
    process = p[0]
    context = p[1]
    process.wait()
    end_time = datetime.datetime.now()
    duration = end_time - context["start_time"]
    os.makedirs(context["exp_base_name"], exist_ok=True)
    txt_log_path = os.path.join(context["exp_base_name"], "log.txt")
    print(
        f"Finished: {context['model']}, {context['transformation']}, {context['search_method']}, duration: {duration}"
    )
