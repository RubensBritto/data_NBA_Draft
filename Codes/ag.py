import numpy as np
import pandas as pd
import math
import statistics
import random
import secrets
import time
from numpy.random import default_rng
import sklearn
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier

# Gloval Variables
POPULATION = []
NEW_POPULATION = []
length_population = 10
length_chromosome = 10
points = []
score_points = []
CROSSOVER_RATE = 75
MUTATION_RATE = 30
good_number = 0

newDict = {}
features = ['GP', 'Min_per', 'Ortg', 'usg', 'eFG', 'TS_per', 'ORB_per', 'DRB_per', 'AST_per', 'TO_per', 'FTM', 'FTA', 'FT_per', 'twoPM', 'twoPA', 'twoP_per', 'TPM', 'TPA', 'TP_per', 'blk_per', 'stl_per', 'ftr', 'adjoe', 'pfr', 'pid', 'ast/tov', 'rimmade', 'rimmade+rimmiss', 'midmade',
            'midmade+midmiss', 'rimmade/(rimmade+rimmiss)', 'midmade/(midmade+midmiss)', 'dunksmade', 'dunksmiss+dunksmade', 'dunksmade/(dunksmade+dunksmiss)', 'drtg', 'adrtg', 'dporpag', 'stops', 'bpm', 'obpm', 'dbpm', 'gbpm', 'mp', 'ogbpm', 'dgbpm', 'oreb', 'dreb', 'treb', 'ast', 'stl', 'blk', 'pts']
ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
       27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]

for i, j in zip(ids, features):
    newDict[i] = j

# AG
def algortimo_genetico(features):
    data = pd.read_csv("output//newDataSet.csv")
    X = data[features]
    y = data["ROUND"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=50)
    dtree = DecisionTreeClassifier().fit(X_train, y_train)
    y_pred = dtree.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)

# Class Chromosome


class Chromosome:
    score = 0

    def __init__(self, schema):
        self.schema = schema

    def __str__(self):
        toString = ''
        for ind in self.schema:
            toString += ind
        return toString


# Selection
def selection(population, new_population):
    merged_list = []
    merged_list.extend(population)
    merged_list.extend(new_population)

    merged_list.sort(key=lambda schema: schema.score, reverse=True)

    return merged_list[:len(POPULATION)]

# CrossOver


def crossOver(population):
    while len(NEW_POPULATION) < len(POPULATION):
        father = population[np.random.randint(0, len(population))].schema
        mother = population[np.random.randint(0, len(population))].schema
        yes = np.random.randint(0, 100)

        if father != mother and yes <= CROSSOVER_RATE:
            child = []
            cut = np.random.randint(1, len(father))
            child.append(father[:cut] + mother[cut:])
            child.append(mother[:cut] + father[cut:])
            for downward in child:
                NEW_POPULATION.append(Chromosome(downward))


def decode(ind):
    resultado = []
    for i in ind:
        for chave, valor in newDict.items():
            if chave == i:
                resultado.append(valor)
    return resultado


def score(population_test):
    for ind in population_test:
        count = 0
        ret = decode(ind.schema)
        ind.score = algortimo_genetico(ret)


# Population

def random():
    rng = default_rng()
    numbers = rng.choice(range(0, 53), size=length_chromosome, replace=False)
    return numbers


def init_population(length_population, length_chromosome):
    for _ in range(length_population):
        array = []
        array.extend(random())
        POPULATION.append(Chromosome(array))


# Main
list_score = []

flag = False
generation = 0
count_aux = 0
POPULATION.clear()
NEW_POPULATION.clear()
init_population(length_population, length_chromosome)

while True:
    score(POPULATION)
    crossOver(POPULATION) 
    score(NEW_POPULATION)
    POPULATION = selection(POPULATION, NEW_POPULATION)
    NEW_POPULATION.clear()

    for ind in POPULATION:
        list_score.append(ind.score)

    if max(list_score) > good_number:
        good_number = max(list_score)
        count_aux += 1

    list_score.clear()

    for ind in POPULATION:
        if ind.score > 0.8 or count_aux == 10:
            flag = True

    if flag:
        print("===================================================================")
        print(
            f'Individuo: {POPULATION[0].schema} e o score dele {POPULATION[0].score} geracao {generation}')
        print("===================================================================")
        score_points.append(POPULATION[0].score)
        points.append(generation)
        break

    generation += 1
