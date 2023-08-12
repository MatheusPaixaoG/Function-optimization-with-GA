import copy, math, random

import params
from evolutive.Individual import *

def step_method_based_individual(features, step):
    individual = None
    if(params.EE["step_method"] == "single"):
        individual = Individual(features, step)
    elif(params.EE["step_method"] == "multi"):
        individual = IndividualMulti(features, step)
    else:
        print("Step method not found")

    return individual

def mutate(population, learning_rate, learning_rate_global):
    new_population = []
    for individual in population:
        # Copying original values
        features = copy.deepcopy(individual.features)
        step = copy.deepcopy(individual.step)

        new_step = None

        # Mutating evolution step
        if (params.EE["step_method"] == "single"):
            new_step = step * math.exp(learning_rate * random.gauss(0,1))
            if new_step < params.EE["mutation_epsilon"]:
                new_step = params.EE["mutation_epsilon"]
        elif (params.EE["step_method"] == "multi"):
            new_step = []
            step_global = learning_rate_global * random.gauss(0,1)
            for i in range(len(step)):
                step_local = learning_rate * random.gauss(0,1)
                
                curr_step = step[i] * math.exp(step_global + step_local)
                if curr_step < params.EE["mutation_epsilon"]:
                    curr_step = params.EE["mutation_epsilon"]
                new_step.append(curr_step)
        else:
            print("This step type was not implemented.")


        # Mutating features
        if(params.EE["step_method"] == "single"):
                new_features = [x + new_step * random.gauss(0,1) for x in features]
        elif (params.EE["step_method"] == "multi"):
            new_features = [features[i] + (new_step[i] * random.gauss(0,1)) for i in range(len(features))]

        mutant_indv = step_method_based_individual(new_features, new_step)

        # Discard bad mutations
        if(mutant_indv.fitness > individual.fitness):
            new_population.append(mutant_indv)
        else:
            new_population.append(individual)
    
    return new_population