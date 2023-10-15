Develop a Python genetic algorithm to classify stock data.

## The data
* The data sets are CSVs of feature vectors of floats and each row has a boolean labels.
* the datasets `train.csv` and `test.csv` are in the `../example_data/` directory. 
* The csv files have no header.
* All but the final column in these files are floating-point feature values
* The final column contains boolean labels (1/0).
* Each row represents a sample of data about a stock for a specific day.
* `train.csv` is used for all of the chromosome evaluation for each generation.
* `test.csv` is used only for evaluation of the best chromosome of the final generation.


## Chromosomes
* Define a Chromosome class
* A chromosome is a set of genes that are all boolean conditions combined via AND.
For example a chromosome with 6 genes must have all 6 be true for the chromosome to resolve to true.
* fail-fast logic can be implemented: if any gene returns false, the overall chromosome returns false.
* When a chromosome is created it will have between 2 and 8 random genes.

### Chromosome Mutation
Mutation on a chromosome can happen in the following ways:
* a gene can be mutated
* a new gene can be added
* a gene can be deleted.
The mutation rate for additions and deletions is 1% - meaning this type of mutation applies to only 1% of children.
The mutation rate for gene mutations is 5%.

## Genes
* Define a Gene class
* A gene has three elements: the index of the feature vector, a lower bound and an upper bound.
* A gene returns true if the value of the feature vector at the specified index is between the lower and upper bound.

### Gene Creation
* When a gene is created, its index is set as a randomly picked index of the feature vector
* two randomly generated numbers between 0.5 and 2.0 are created.  The lesser is set as the lower bound, 
the other the upper bound

### Gene mutation
Mutation on a gene can happen in the following way:
* the index is shifted up or down by 1.  if this results in being index out of bounds, then the mutation is not applied.
* the lower bound is multiplied by a random value between .95 and .1.05
* similarly for the upper bound

## Evolving
* Each generation will have 10000 chromosomes.
* run for 100 generations.
* Evolving will be done on the data in `train.csv`
* To evaluate a generation, randomly select a subset of 10% of the test set and evaluate on that.
* The fitness function for a chromosome is (the squared number of rows correctly labeled) / (the number of rows labeled as true)
* score each chromosome with the fitness function
* rank the chromosomes by fitness
* for the top chromosome, find the confusion matrix variables and print the number of true positives and the precision.
* Remove the lower 95% of chromosomes
* Use sexual reproduction of the remaining chromomes to replace the removed chromosomes.

Sexual recombination is handled in the following way:
* Child is made of a combination of the genes of parents A and B.
* Child starts out with no genes.
* For each gene in parent A, add that gene to child with a random probability of 50%
* Do the same for parent B.
* Apply mutation to the child.

## Final evaluation
run the best performing chromosome of the final generation on `test.csv`
Calculate the confusion matrix variables then use them to 
execute `print_statistics(tp=TP, fp=FP, tn=TN, fn=FN)`.  load this function with `from common import print_statistics`.

### Note
- Comment complex or unclear code sections adequately.
- Adopt a "less is more" approach when coding.
- Create this as one continuous block of code
- Have all imports be declared at the beginning
- Have all global constants be declared next
