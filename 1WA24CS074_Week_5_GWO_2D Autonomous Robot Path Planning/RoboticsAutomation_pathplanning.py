import numpy as np


def calculate_fitness(position, start, goal, obstacles, safe_margin=0.25):
    """Calculates total path distance plus penalties for hitting obstacles."""
    waypoints = position.reshape(-1, 2)
    path = np.vstack((start, waypoints, goal))

    total_length = 0.0
    penalty = 0.0

    # Check every straight segment along the path
    for k in range(len(path) - 1):
        p_start = path[k]
        p_end = path[k + 1]
        total_length += float(np.linalg.norm(p_end - p_start))

        # Vectorized collision check across 30 points along the segment
        steps = np.linspace(0.0, 1.0, 30)[:, None]
        segment_points = p_start + steps * (p_end - p_start)

        for ox, oy, radius in obstacles:
            center = np.array([ox, oy])
            distances = np.linalg.norm(segment_points - center, axis=1)
            violations = (radius + safe_margin) - distances
            # Add penalty wherever a point enters the obstacle's safety zone
            bad_points = violations[violations > 0]
            if len(bad_points) > 0:
                penalty += 1500.0 * float(np.sum(bad_points))

    return total_length + penalty


def run_gwo_planner(
    start, goal, obstacles, num_waypoints=3, map_limit=10.0, pack_size=35, iterations=200
):
    """Runs Grey Wolf Optimization to find the shortest collision-free trajectory."""
    dimensions = num_waypoints * 2
    pack = np.random.uniform(0.0, map_limit, size=(pack_size, dimensions))

    alpha_pos = np.zeros(dimensions)
    beta_pos = np.zeros(dimensions)
    delta_pos = np.zeros(dimensions)

    alpha_score = float("inf")
    beta_score = float("inf")
    delta_score = float("inf")

    for step in range(iterations):
        # 1. Evaluate all wolves and rank the top three leaders
        for i in range(pack_size):
            pack[i] = np.clip(pack[i], 0.0, map_limit)
            score = calculate_fitness(pack[i], start, goal, obstacles)

            if score < alpha_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = alpha_score, alpha_pos.copy()
                alpha_score, alpha_pos = score, pack[i].copy()
            elif score < beta_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = score, pack[i].copy()
            elif score < delta_score:
                delta_score, delta_pos = score, pack[i].copy()

        # Print live progress every 50 iterations
        if (step + 1) % 50 == 0 or step == 0:
            print(f"Iteration {step + 1:3d}/{iterations} | Best Path Cost: {alpha_score:.4f}")

        # 2. Update wolf positions using vectorized GWO equations
        a = 2.0 * (1.0 - step / iterations)

        r1 = np.random.rand(pack_size, dimensions)
        r2 = np.random.rand(pack_size, dimensions)
        A1 = 2.0 * a * r1 - a
        C1 = 2.0 * r2
        X1 = alpha_pos - A1 * np.abs(C1 * alpha_pos - pack)

        r1 = np.random.rand(pack_size, dimensions)
        r2 = np.random.rand(pack_size, dimensions)
        A2 = 2.0 * a * r1 - a
        C2 = 2.0 * r2
        X2 = beta_pos - A2 * np.abs(C2 * beta_pos - pack)

        r1 = np.random.rand(pack_size, dimensions)
        r2 = np.random.rand(pack_size, dimensions)
        A3 = 2.0 * a * r1 - a
        C3 = 2.0 * r2
        X3 = delta_pos - A3 * np.abs(C3 * delta_pos - pack)

        pack = (X1 + X2 + X3) / 3.0

    final_waypoints = alpha_pos.reshape(-1, 2)
    best_trajectory = np.vstack((start, final_waypoints, goal))
    return best_trajectory, alpha_score


if __name__ == "__main__":
    start_coord = np.array([0.0, 0.0])
    goal_coord = np.array([10.0, 10.0])

    # Obstacles formatted as (x_center, y_center, radius)
    env_obstacles = [
        (2.5, 2.5, 1.0),
        (5.0, 5.5, 1.4),
        (7.5, 3.0, 1.1),
        (3.5, 7.5, 1.0),
        (7.8, 7.8, 0.9),
    ]

    print("Starting GWO Autonomous Path Planner...\n")
    trajectory, final_cost = run_gwo_planner(
        start=start_coord,
        goal=goal_coord,
        obstacles=env_obstacles,
        num_waypoints=3,
        pack_size=35,
        iterations=200,
    )

    print("\n--- Final Collision-Free Trajectory ---")
    for idx, point in enumerate(trajectory):
        x_val, y_val = point
        if idx == 0:
            name = "Start"
        elif idx == len(trajectory) - 1:
            name = "Goal"
        else:
            name = f"Waypoint {idx}"
        print(f"  {name:10s} -> X: {x_val:6.3f}, Y: {y_val:6.3f}")

    print(f"\nOptimal Path Length: {final_cost:.4f} units")
