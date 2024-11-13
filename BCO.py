import random
import numpy as np
import time


def BCO(population, cost_func, lb, ub, max_iter):
    """
    Border Collie Optimization (BCO)
    """
    population_size, dim = population.shape[0], population.shape[1]
    bounds = [lb, ub]
    num_variables = dim
    Convergence = np.zeros(max_iter)
    # Initialize the fitness values
    fitness_values = np.zeros(population_size)
    for i in range(population_size):
        fitness_values[i] = cost_func(population[i])


    ct = time.time()
    # Main loop
    for iteration in range(max_iter):
        for i in range(population_size):
            print(iteration, i)
            # Generate random indices for the border to interact with
            indices = list(range(population_size))
            indices.remove(i)
            random.shuffle(indices)

            # Select two random individuals from the population
            firefly1 = population[i]
            firefly2 = population[indices[0]]

            # Calculate the attractiveness between the border
            beta = 1.0  # Attraction coefficient
            distance = np.linalg.norm(firefly1 - firefly2)
            attractiveness = np.exp(-beta * distance)

            # Move the firefly towards the brighter one
            firefly1 += attractiveness * (firefly2 - firefly1)

            # Update the fitness value of the border
            fitness_values[i] = cost_func(firefly1)

        # Sort the population based on fitness values
        sorted_indices = np.argsort(fitness_values)
        population = population[sorted_indices]
        fitness_values = fitness_values[sorted_indices]
        Convergence[iteration] = np.min(fitness_values)
    ct = time.time() - ct
    # Return the best solution and its fitness value
    best_solution = population[0]
    best_fitness = np.min(fitness_values)

    return best_fitness, Convergence, best_solution,  ct
