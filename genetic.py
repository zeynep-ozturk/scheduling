#read the data
import pandas as pd
import numpy as np
import time
car1 = pd.read_csv('fssp-data-3instances.txt', delimiter=r'\s+', nrows=8, skiprows=8, names=['m0', 'p0', 'm1', 'p1', 'm2', 'p2', 'm3', 'p3'])
reC05 = pd.read_csv('fssp-data-3instances.txt', delimiter=r'\s+', nrows=13, skiprows=19, names=['m0', 'p0', 'm1', 'p1', 'm2', 'p2', 'm3', 'p3', 'm4', 'p4'])
reC09 = pd.read_csv('fssp-data-3instances.txt', delimiter=r'\s+', nrows=20, skiprows=35, names=['m0', 'p0', 'm1', 'p1', 'm2', 'p2', 'm3', 'p3', 'm4', 'p4'])

car1.drop(['m0', 'm1', 'm2', 'm3'], inplace=True, axis=1)
reC05.drop(['m0', 'm1', 'm2', 'm3', 'm4'], inplace=True, axis=1)
reC09.drop(['m0', 'm1', 'm2', 'm3', 'm4'], inplace=True, axis=1)

car1 = np.array(car1)
reC05 = np.array(reC05)
reC09 = np.array(reC09)

car1_opt  = 4534
reC05_opt = 920
reC09_opt = 1302

def calculate_makespan(sol,processingTimes):
    solution = np.argsort(sol) #convert random key representation to permutation representation
    n_jobs,n_machines = processingTimes.shape
    completion_times = np.zeros(processingTimes.shape)
    completion_times[0, :]=np.cumsum(processingTimes[solution[0],:])
    completion_times[:, 0] = np.cumsum(processingTimes[solution, 0])
    for j in np.arange(1,n_jobs):
        for m in np.arange(1,n_machines):
            completion_times[j,m] = processingTimes[solution[j], m] + max(completion_times[j,m-1],completion_times[j-1,m])
    return completion_times[n_jobs-1,n_machines-1]

#initialization
def initialization(seed,n_pop,n_jobs):
    # random key representation
    fixed_rng = np.random.RandomState(seed=seed)
    unsorted = fixed_rng.uniform(0, 1, size=(n_pop, n_jobs))
    #x = np.argsort(unsorted, axis=1)
    return unsorted


#parent selection
def selection(fitness,parent_size,seed):
    selection_prob = (max(fitness) - fitness) / sum(max(fitness) - fitness)
    selection_cum_prob = np.cumsum(selection_prob)
    #roulette wheel selection
    fixed_rng = np.random.RandomState(seed=seed)
    randomnumbers = fixed_rng.uniform(0, 1,size=parent_size)
    return [np.min(np.where(randomnumbers[i]<=selection_cum_prob)) for i in range(parent_size)]

def crossOver(parents): #1 point crossover
    parent1, parent2 = parents[0], parents[1]
    crossOver_pt = int(np.random.randint(1,len(parent1),1))
    child = np.hstack((parent1[:crossOver_pt],parent2[crossOver_pt:]))
    return child

def mutation(child): #swap
    a, b = np.random.randint(0, len(child), size=2)
    child[a], child[b] = child[b], child[a]
    return child

def replacement(pop,fitness,child,seed):
    replacement_size = 1
    replacement_prob = (fitness - min(fitness)) / sum(fitness - min(fitness))
    replacement_cum_prob = np.cumsum(replacement_prob)
    #roulette wheel like replacement
    fixed_rng = np.random.RandomState(seed=seed)
    randomnumbers = fixed_rng.uniform(0, 1,size=replacement_size)
    parents_tobereplaced = [np.min(np.where(randomnumbers[i] <= replacement_cum_prob)) for i in range(replacement_size)]
    pop[parents_tobereplaced] = child
    fitness[parents_tobereplaced] = calculate_makespan(np.argsort(child), processingTimes)
    return pop, fitness

processingTimes=car1
optimal=car1_opt
n_pop = 100
parent_size=2
crossOverProb = 0.8
mutationProb = 0.8
def genetic_algorithm(processingTimes,optimal,n_pop):
    start = time.time()
    n_jobs = len(processingTimes)
    ##initialization
    iter=1
    population = initialization(seed=42,n_pop=n_pop,n_jobs=n_jobs)
    while(True):
        ##seed
        seed=iter
        ##calculate fitness
        fitness = np.apply_along_axis(calculate_makespan, 1, population, processingTimes)
        incumbent_obj  = np.min(fitness)
        incumbent_soln = np.argsort(population[np.argmin(fitness)])
        incumbent_gap = (incumbent_obj - optimal) / optimal
        ##selection
        selected_indices = selection(fitness, parent_size, seed)
        parents = population[selected_indices]
        ##cross-over
        if np.random.rand()<=crossOverProb:
            child = crossOver(parents)
        else:
            child = parents[np.random.randint(0,2)]
        ##mutation
        if np.random.rand()<=mutationProb:
            child = mutation(child)
        ##replacement
        new_population, new_fitness = replacement(population, fitness, child, seed)
        population = new_population.copy()
        fitness = new_fitness.copy()
        ##change incumbent if there is improvement
        if incumbent_obj > np.min(fitness):
            incumbent_obj = np.min(fitness)
            incumbent_soln = np.argsort(population[np.argmin(fitness)])
            incumbent_gap = (incumbent_obj - optimal) / optimal
        ##stopping criteria
        if incumbent_gap<0.00001:
            average_obj = np.mean(fitness)
            break
        iter+=1
    end = time.time()
    time_elapsed = end-start
    return incumbent_soln, optimal, incumbent_obj, average_obj, incumbent_gap, time_elapsed

print(genetic_algorithm(processingTimes,optimal,n_pop))
