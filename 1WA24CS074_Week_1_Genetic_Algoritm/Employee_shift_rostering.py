import random
import copy
import csv

# --- CONFIGURATION ---
EMPLOYEES = ["Alice", "Bob", "Charlie", "David", "Eva"]

# 14-day schedule cycle (2 full weeks)
DAYS = [
    "Week1_Mon", "Week1_Tue", "Week1_Wed", "Week1_Thu", "Week1_Fri", "Week1_Sat", "Week1_Sun",
    "Week2_Mon", "Week2_Tue", "Week2_Wed", "Week2_Thu", "Week2_Fri", "Week2_Sat", "Week2_Sun"
]
SHIFTS = ["Morning", "Afternoon", "Night"]

POPULATION_SIZE = 200
GENERATIONS = 500  
TOURNAMENT_SIZE = 5
CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.20

# 14 days * 3 shifts = 42 total unique shift slots to fill
TOTAL_SLOTS = len(DAYS) * len(SHIFTS)

# Map every slot index directly to its specific Day and Shift type
SLOT_MAP = []
for day in DAYS:
    for shift in SHIFTS:
        SLOT_MAP.append((day, shift))

# --- GENETIC ALGORITHM ENGINE ---

def create_chromosome():
    """Generates a random initial roster string."""
    return [random.choice(EMPLOYEES) for _ in range(TOTAL_SLOTS)]

def calculate_fitness(chromosome):
    """
    Evaluates the schedule quality. 
    Ideal fitness is 0. Penalties subtract points.
    """
    penalty = 0
    
    # 1. HARD CONSTRAINT: Max 1 shift per employee on any single calendar day
    daily_workload = {day: {emp: 0 for emp in EMPLOYEES} for day in DAYS}
    for slot_idx, employee in enumerate(chromosome):
        day, _ = SLOT_MAP[slot_idx]
        daily_workload[day][employee] += 1

    for day in DAYS:
        for employee in EMPLOYEES:
            shifts_today = daily_workload[day][employee]
            if shifts_today > 1:
                penalty += (shifts_today - 1) * 100  # Heavy penalty for double-booking
                
    # 2. SOFT CONSTRAINT: Rest periods (Anti-Clopening)
    # Checks if an employee works a Night shift followed immediately by a Morning shift
    for idx in range(TOTAL_SLOTS - 1):
        _, current_shift = SLOT_MAP[idx]
        _, next_shift = SLOT_MAP[idx + 1]
        
        if current_shift == "Night" and next_shift == "Morning":
            if chromosome[idx] == chromosome[idx + 1]:
                penalty += 30  # Penalty to prevent back-to-back shifts
                
    return -penalty

def tournament_selection(population, fitnesses):
    """Selects the best roster out of a small random pool."""
    best_idx = random.randint(0, len(population) - 1)
    for _ in range(TOURNAMENT_SIZE - 1):
        idx = random.randint(0, len(population) - 1)
        if fitnesses[idx] > fitnesses[best_idx]:
            best_idx = idx
    return copy.deepcopy(population[best_idx])

def crossover(parent1, parent2):
    """Swaps data between two parent schedules at a random breakpoint."""
    if random.random() < CROSSOVER_RATE:
        split = random.randint(1, TOTAL_SLOTS - 1)
        return parent1[:split] + parent2[split:], parent2[:split] + parent1[split:]
    return copy.deepcopy(parent1), copy.deepcopy(parent2)

def mutate(chromosome):
    """Randomly alters shift assignments to maintain genetic diversity."""
    for idx in range(TOTAL_SLOTS):
        if random.random() < MUTATION_RATE:
            chromosome[idx] = random.choice(EMPLOYEES)
    return chromosome

def run_genetic_algorithm():
    """Runs the evolutionary loop until completed or a optimal solution is found."""
    population = [create_chromosome() for _ in range(POPULATION_SIZE)]
    
    for generation in range(GENERATIONS):
        fitnesses = [calculate_fitness(chrom) for chrom in population]
        max_fitness = max(fitnesses)
        
        # Stop early only if it hits a perfect score of 0 (no conflicts, no clopenings)
        if max_fitness == 0:
            break
            
        next_pop = []
        while len(next_pop) < POPULATION_SIZE:
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            c1, c2 = crossover(p1, p2)
            next_pop.extend([mutate(c1), mutate(c2)])
        population = next_pop[:POPULATION_SIZE]

    final_fitnesses = [calculate_fitness(chrom) for chrom in population]
    best_chrom = population[final_fitnesses.index(max(final_fitnesses))]
    return best_chrom, max(final_fitnesses)

# --- CLEAN MATRIX TABLE EXPORT ---

def save_to_matrix_csv(chromosome, filepath):
    """Saves the flat roster array into an organized grid format."""
    # Structure data as a dictionary: { Day: { Shift: Employee } }
    roster_matrix = {day: {} for day in DAYS}
    for idx, employee in enumerate(chromosome):
        day, shift = SLOT_MAP[idx]
        roster_matrix[day][shift] = employee

    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Table columns setup
        writer.writerow(["Day"] + SHIFTS)
        
        # Add 14 organized rows to the spreadsheet
        for day in DAYS:
            row_data = [
                day, 
                roster_matrix[day]["Morning"], 
                roster_matrix[day]["Afternoon"], 
                roster_matrix[day]["Night"]
            ]
            writer.writerow(row_data)

def main():
    print("Evolution engine initializing...")
    print("Generating 3 distinct 14-day conflict-free alternate rosters...")
    
    # Creates three separate files with completely unique distributions
    for option_num in range(1, 4):
        best_roster, score = run_genetic_algorithm()
        filename = f"roster_table_option_{option_num}.csv"
        save_to_matrix_csv(best_roster, filename)
        print(f" -> Option {option_num} completed (Fitness Score: {score}). Saved to {filename}")
        
    print("\nAll files successfully exported. Check your current directory!")

if __name__ == "__main__":
    main()
