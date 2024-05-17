import textattack
import os, sys

current_dir = os.path.dirname(os.path.realpath(__file__))
constraint_dir = os.path.normpath(os.path.join(current_dir, os.pardir))
transformation_dir = os.path.normpath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(current_dir)
sys.path.append(constraint_dir)
sys.path.append(transformation_dir)
from transformation import TRANSFORMATION
from constraint import CONSTRAINTS
from recipes.word_swap_wordnet.strict.tabu_dbscan.tabu_dbscan import (
    EmbeddingBasedTabuSearch,
)


def Attack(model):
    goal_function = textattack.goal_functions.UntargetedClassification(model)
    search_method = EmbeddingBasedTabuSearch()
    transformation = TRANSFORMATION
    constraints = CONSTRAINTS
    return textattack.Attack(goal_function, constraints, transformation, search_method)
