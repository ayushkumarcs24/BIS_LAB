import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Obstacle Definition ---
class CircularObstacle:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

    def is_segment_hitting(self, p1, p2):
        d = p2 - p1
        f = p1 - np.array([self.x, self.y])
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - self.radius**2
        
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            return False
        
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)
        
        if (0 <= t1 <= 1) or (0 <= t2 <= 1):
            return True
        if t1 < 0 and t2 > 1:
            return True
        return False

# --- Fitness Calculation ---
def calculate_fitness(waypoints, start, goal, obstacles):
    full_path = np.vstack([start, waypoints, goal])
    total_length = 0.0
    penalty = 0.0
    HEAVY_PENALTY = 1e6
    
    for i in range(len(full_path) - 1):
        p1 = full_path[i]
        p2 = full_path[i + 1]
        total_length += np.linalg.norm(p2 - p1)
        
        for obs in obstacles:
            if obs.is_segment_hitting(p1, p2):
                penalty += HEAVY_PENALTY
    return total_length + penalty

# --- Optimization Parameters ---
start_pos = np.array([0.0, 0.0])
goal_pos = np.array([10.0, 10.0])
obs_list = [CircularObstacle(x=5.0, y=5.0, radius=2.0)]
num_waypoints = 4
num_particles = 20
max_iter = 50

# Initialize particles
X = np.random.uniform(0, 10, (num_particles, num_waypoints, 2))
V = np.random.uniform(-0.5, 0.5, (num_particles, num_waypoints, 2))
P_best = np.copy(X)
P_best_fitness = np.array([calculate_fitness(X[i], start_pos, goal_pos, obs_list) for i in range(num_particles)])

g_best_idx = np.argmin(P_best_fitness)
G_best = np.copy(P_best[g_best_idx])
G_best_fitness = P_best_fitness[g_best_idx]

w, c1, c2 = 0.5, 1.5, 1.5

# --- Matplotlib Plot Setup ---
fig, ax = plt.subplots(figsize=(7, 7))

def update(frame):
    global X, V, P_best, P_best_fitness, G_best, G_best_fitness
    ax.clear()
    
    # 1. Draw Map Elements
    ax.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start (0,0)')
    ax.plot(goal_pos[0], goal_pos[1], 'ro', markersize=10, label='Goal (10,10)')
    circle = plt.Circle((5.0, 5.0), 2.0, color='gray', alpha=0.6, label='Obstacle')
    ax.add_patch(circle)
    
    # 2. Update PSO Math Engine
    for i in range(num_particles):
        r1, r2 = np.random.rand(2)
        V[i] = w * V[i] + c1 * r1 * (P_best[i] - X[i]) + c2 * r2 * (G_best - X[i])
        X[i] += V[i]
        
        # Keep inside graph limits
        X[i] = np.clip(X[i], 0, 10)
        
        current_fitness = calculate_fitness(X[i], start_pos, goal_pos, obs_list)
        if current_fitness < P_best_fitness[i]:
            P_best[i] = np.copy(X[i])
            P_best_fitness[i] = current_fitness
            if current_fitness < G_best_fitness:
                G_best = np.copy(X[i])
                G_best_fitness = current_fitness

    # 3. Plot Current Candidate Paths (Faded Blue Lines)
    for i in range(num_particles):
        p = np.vstack([start_pos, X[i], goal_pos])
        ax.plot(p[:, 0], p[:, 1], color='skyblue', alpha=0.3, linewidth=1)
        ax.scatter(X[i][:, 0], X[i][:, 1], color='blue', alpha=0.2, s=15)

    # 4. Plot Global Best Path Found So Far (Thick Red Line)
    best_p = np.vstack([start_pos, G_best, goal_pos])
    ax.plot(best_p[:, 0], best_p[:, 1], color='crimson', linewidth=3, label='Global Best Path')
    ax.scatter(G_best[:, 0], G_best[:, 1], color='crimson', s=40, zorder=5)
    
    # Plot formatting
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)
    ax.set_title(f"PSO Path Planning — Iteration {frame+1}/{max_iter}\nBest Fitness (Length): {G_best_fitness:.4f}")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left')

# Run the animation loop
ani = animation.FuncAnimation(fig, update, frames=max_iter, repeat=False, interval=200)
plt.show()
