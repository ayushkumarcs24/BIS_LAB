
import numpy as np
import random
import matplotlib.pyplot as plt

# ---------------------------------------
# CLOUD COMPUTING PARAMETERS
# ---------------------------------------

num_jobs = 10
num_vms = 4
num_ants = 20
iterations = 100

# Job lengths (Million Instructions)
jobs = [1200, 2500, 1800, 3000, 1500,
        2200, 1000, 2800, 1700, 2000]

# VM processing capacities (MIPS)
vms = [100, 150, 200, 250]

# ACO parameters
alpha = 1.0
beta = 2.0
evaporation = 0.5
Q = 100

# Fitness weights
w1 = 0.7
w2 = 0.3

# ---------------------------------------
# INITIALIZATION
# ---------------------------------------

pheromone = np.ones((num_jobs, num_vms))

best_solution = None
best_fitness = float('inf')

fitness_history = []

# ---------------------------------------
# FITNESS FUNCTION
# ---------------------------------------

def calculate_fitness(solution):

    vm_load = np.zeros(num_vms)

    for job, vm in enumerate(solution):
        execution_time = jobs[job] / vms[vm]
        vm_load[vm] += execution_time

    makespan = max(vm_load)

    average_load = np.mean(vm_load)

    if average_load == 0:
        imbalance = 0
    else:
        imbalance = np.std(vm_load) / average_load

    fitness = w1 * makespan + w2 * imbalance * makespan

    return fitness, makespan, vm_load, imbalance


# ---------------------------------------
# HEURISTIC INFORMATION
# ---------------------------------------

def calculate_heuristic(job):

    heuristic = []

    for vm in range(num_vms):

        execution_time = jobs[job] / vms[vm]

        heuristic.append(1 / (execution_time + 0.001))

    return np.array(heuristic)


# ---------------------------------------
# SELECT VM USING ACO
# ---------------------------------------

def select_vm(job, vm_load):

    heuristic = calculate_heuristic(job)

    probabilities = []

    for vm in range(num_vms):

        # Consider processing capacity and current VM workload
        load_factor = 1 / (1 + vm_load[vm])

        probability = (
            (pheromone[job][vm] ** alpha) *
            (heuristic[vm] ** beta) *
            load_factor
        )

        probabilities.append(probability)

    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()

    selected_vm = np.random.choice(
        num_vms, p=probabilities
    )

    return selected_vm


# ---------------------------------------
# ANT COLONY OPTIMIZATION
# ---------------------------------------

for iteration in range(iterations):

    all_solutions = []
    all_fitness = []

    for ant in range(num_ants):

        solution = []
        vm_load = np.zeros(num_vms)

        # Schedule every job
        for job in range(num_jobs):

            selected_vm = select_vm(job, vm_load)

            solution.append(selected_vm)

            execution_time = jobs[job] / vms[selected_vm]

            vm_load[selected_vm] += execution_time

        fitness, makespan, loads, imbalance = (
            calculate_fitness(solution)
        )

        all_solutions.append(solution)
        all_fitness.append(fitness)

        # Update global best
        if fitness < best_fitness:

            best_fitness = fitness
            best_solution = solution.copy()

    # -----------------------------------
    # PHEROMONE EVAPORATION
    # -----------------------------------

    pheromone *= (1 - evaporation)

    # -----------------------------------
    # PHEROMONE DEPOSIT
    # -----------------------------------

    for ant in range(num_ants):

        solution = all_solutions[ant]
        fitness = all_fitness[ant]

        deposit = Q / (fitness + 0.001)

        for job, vm in enumerate(solution):

            pheromone[job][vm] += deposit

    fitness_history.append(best_fitness)

    print(
        f"Iteration {iteration + 1}: "
        f"Best Fitness = {best_fitness:.4f}"
    )


# ---------------------------------------
# FINAL RESULTS
# ---------------------------------------

fitness, makespan, vm_load, imbalance = (
    calculate_fitness(best_solution)
)

print("\n========== ACO CLOUD SCHEDULING RESULTS ==========")

print("\nJob Scheduling Allocation:")

for job, vm in enumerate(best_solution):

    execution_time = jobs[job] / vms[vm]

    print(
        f"Job J{job + 1} --> VM{vm + 1} "
        f"| Execution Time = {execution_time:.2f}"
    )

print("\nVM Workload Distribution:")

for vm in range(num_vms):

    print(
        f"VM{vm + 1}: {vm_load[vm]:.2f} seconds"
    )

print("\nPerformance Metrics:")

print(f"Makespan: {makespan:.4f}")
print(f"Load Imbalance: {imbalance:.4f}")
print(f"Best Fitness: {fitness:.4f}")

# ---------------------------------------
# GRAPH 1: CONVERGENCE
# ---------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, iterations + 1),
    fitness_history,
    color='blue'
)

plt.xlabel("Iterations")
plt.ylabel("Best Fitness")
plt.title("ACO Convergence for Cloud Job Scheduling")
plt.grid(True)

plt.show()


# ---------------------------------------
# GRAPH 2: VM LOAD BALANCING
# ---------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    [f"VM{i+1}" for i in range(num_vms)],
    vm_load,
    color='green'
)

plt.xlabel("Virtual Machines")
plt.ylabel("Total Execution Time")
plt.title("ACO Load Balancing in Cloud Computing")
plt.grid(axis='y')

plt.show()
