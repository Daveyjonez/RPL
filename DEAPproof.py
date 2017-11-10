#Author: David Owens
#Date: 11/6/17

import random
import matplotlib.pyplot as plt
from deap import base
from deap import creator
from deap import tools

NON_CONST_VARS = 2
GAMMA = 1.4

#Define the fitnessMax function as a maximization function
#i.e. Maximize the compressor ratio
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#creator.create("FitnessMulti", base.Fitness, weights=(1.0,-1.0,))

#Create the Compressor class which derives from list and assign the fitnessMax
#Do not assign entryTemp & exitTemp as class vars because those will change
#each instance of class, set min and maxTemp to be class vars since we want
#all temps to operate within that range
creator.create("Compressor", list, fitness=creator.FitnessMax)

#Calculate CPR, the larger the better
def evalCompressor(Compressor):
    cpr = (Compressor[1] / Compressor[0]) ** (GAMMA / (GAMMA-1))
    return cpr,

#Register the toolbox as toolbox, standard line, should be included in every
#DEAP program
toolbox = base.Toolbox()

#Attribute generator
toolbox.register("attr_rand", random.randint, 400, 401)

#Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Compressor,
    toolbox.attr_rand, NON_CONST_VARS)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#Register the fitness function in toolbox
toolbox.register("evaluate", evalCompressor)

#Register mating function in toolbox
toolbox.register("mate", tools.cxTwoPoint)

#Register mutate function in toolbox
toolbox.register("mutate", tools.mutUniformInt, low=250, up=600, indpb=0.05)

#Register tournament function in toolbox
toolbox.register("select", tools.selTournament, tournsize=3)

#Main method
def main():
    random.seed(64)

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=500)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    #
    # NGEN  is the number of generations for which the
    #       evolution runs
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i Compressors" % len(pop))

    # Begin the evolution
    for g in range(NGEN):
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i Compressors" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]


        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        print("  Best compressor temps %s" % tools.selBest(pop, 1)[0])

        plt.plot(g, mean, 'bo')

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, with a CPR of %s" 
          % (best_ind, best_ind.fitness.values))

    plt.xlabel("Generation #")
    plt.ylabel("Mean fitness (CPR)")
    plt.title("Evolutionary Algorithm: Compressor", fontsize=20)
    plt.show()

if __name__ == "__main__":
    main()
