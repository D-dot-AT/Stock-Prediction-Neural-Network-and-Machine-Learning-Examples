# This WIP.  It does not perform well, the winning chromosome essentially marks all tests as positive.
# I suspect the issue is with the fitness function.  It needs to reward both for high precision as well
# as high number of true positives.

import random
import pandas as pd
from common import print_statistics

# Global Constants
TRAIN_CSV = "../example_data/train.csv"
TEST_CSV = "../example_data/test.csv"
GENES_MIN = 2
GENES_MAX = 8
GENE_BOUND_MIN = 0.5
GENE_BOUND_MAX = 2.0
GENE_MUTATION_RATE = 0.05
CHROMOSOME_MUTATION_ADD_RATE = 0.01
CHROMOSOME_MUTATION_DELETE_RATE = 0.01
POPULATION_SIZE = 10000
GENERATIONS = 100
SAMPLE_SIZE = 0.1


class Gene:
    def __init__(self, feature_count):
        self.index = random.randint(0, feature_count - 1)
        self.lower_bound, self.upper_bound = sorted([random.uniform(GENE_BOUND_MIN, GENE_BOUND_MAX) for _ in range(2)])

    def mutate(self, feature_count):
        self.index = min(max(self.index + random.choice([-1, 1]), 0), feature_count - 1)
        self.lower_bound *= random.uniform(0.95, 1.05)
        self.upper_bound *= random.uniform(0.95, 1.05)

    def evaluate(self, feature_vector):
        return self.lower_bound <= feature_vector[self.index] <= self.upper_bound


class Chromosome:
    def __init__(self, feature_count):
        self.genes = [Gene(feature_count) for _ in range(random.randint(GENES_MIN, GENES_MAX))]

    def mutate(self, feature_count):
        for gene in self.genes:
            if random.random() < GENE_MUTATION_RATE:
                gene.mutate(feature_count)

        if random.random() < CHROMOSOME_MUTATION_ADD_RATE:
            self.genes.append(Gene(feature_count))

        if random.random() < CHROMOSOME_MUTATION_DELETE_RATE and len(self.genes) > GENES_MIN:
            self.genes.pop(random.randint(0, len(self.genes) - 1))

    def evaluate(self, feature_vector):
        return all(gene.evaluate(feature_vector) for gene in self.genes)


def main():
    train_data = pd.read_csv(TRAIN_CSV, header=None)
    test_data = pd.read_csv(TEST_CSV, header=None)

    feature_count = train_data.shape[1] - 1
    population = [Chromosome(feature_count) for _ in range(POPULATION_SIZE)]

    for generation in range(GENERATIONS):
        sample = train_data.sample(frac=SAMPLE_SIZE)
        features = sample.iloc[:, :-1].values
        labels = sample.iloc[:, -1].values

        scores = []
        for chromosome in population:
            tp = tn = fp = fn = 0
            for feature_vector, label in zip(features, labels):
                prediction = chromosome.evaluate(feature_vector)
                if prediction:
                    if label == 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if label == 0:
                        tn += 1
                    else:
                        fn += 1

            if tp + fp == 0:
                precision = 0
            else:
                precision = tp / (tp + fp)

            if precision == 0:
                fitness = 0
            else:
                fitness = (tp ** 2) / (tp + fp)

            scores.append((fitness, tp, precision, chromosome))

        scores.sort(reverse=True, key=lambda x: x[0])
        top_chromosome = scores[0]
        print(f"Generation {generation}: TP={top_chromosome[1]}, Precision={top_chromosome[2]}")

        survivors = [x[3] for x in scores[:POPULATION_SIZE // 20]]

        population = survivors.copy()
        while len(population) < POPULATION_SIZE:
            parent_a, parent_b = random.sample(survivors, 2)
            child = Chromosome(feature_count)
            child.genes = []
            for gene in parent_a.genes:
                if random.random() < 0.5:
                    child.genes.append(gene)
            for gene in parent_b.genes:
                if random.random() < 0.5:
                    child.genes.append(gene)
            child.mutate(feature_count)
            population.append(child)

    best_chromosome = top_chromosome[3]
    features = test_data.iloc[:, :-1].values
    labels = test_data.iloc[:, -1].values
    tp = tn = fp = fn = 0
    for feature_vector, label in zip(features, labels):
        prediction = best_chromosome.evaluate(feature_vector)
        if prediction:
            if label == 1:
                tp += 1
            else:
                fp += 1
        else:
            if label == 0:
                tn += 1
            else:
                fn += 1

    print_statistics(tp=tp, fp=fp, tn=tn, fn=fn)


if __name__ == "__main__":
    main()
