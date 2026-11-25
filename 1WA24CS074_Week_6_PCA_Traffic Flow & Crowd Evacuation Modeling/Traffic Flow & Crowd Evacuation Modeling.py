import time
import numpy as np


def compute_floor_field(grid):
    """Computes a static distance field where every cell stores its distance to the nearest exit."""
    rows, cols = grid.shape
    floor_field = np.full((rows, cols), np.inf)

    # Find all exit cells (value == 2) and set their distance to 0
    exits = np.argwhere(grid == 2)
    queue = []
    for r, c in exits:
        floor_field[r, c] = 0.0
        queue.append((r, c))

    # 8-directional Moore neighborhood offsets (dr, dc, movement_cost)
    directions = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, 1.414),
        (-1, 1, 1.414),
        (1, -1, 1.414),
        (1, 1, 1.414),
    ]

    # Breadth-First propagation to compute shortest paths around walls
    head = 0
    while head < len(queue):
        r, c = queue[head]
        head += 1

        for dr, dc, cost in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr, nc] != -1:  # Ignore walls
                    new_dist = floor_field[r, c] + cost
                    if new_dist < floor_field[nr, nc]:
                        floor_field[nr, nc] = new_dist
                        queue.append((nr, nc))

    return floor_field


def initialize_room(rows=12, cols=20, num_people=35, seed=42):
    """Creates a 2D room grid with outer walls, an internal obstacle, two exits, and random people."""
    np.random.seed(seed)
    grid = np.zeros((rows, cols), dtype=int)

    # Set outer walls (-1)
    grid[0, :] = -1
    grid[-1, :] = -1
    grid[:, 0] = -1
    grid[:, -1] = -1

    # Add an internal pillar/wall obstacle
    grid[4:8, 10] = -1

    # Place two exit doors (2) on the right wall
    grid[3, -1] = 2
    grid[8, -1] = 2

    # Randomly place pedestrians (1) on empty floor cells (0)
    empty_cells = np.argwhere(grid == 0)
    chosen_indices = np.random.choice(
        len(empty_cells), size=num_people, replace=False
    )
    for idx in chosen_indices:
        r, c = empty_cells[idx]
        grid[r, c] = 1

    return grid


def step_parallel_evacuation(grid, floor_field):
    """Executes one synchronous parallel update step across all pedestrians."""
    rows, cols = grid.shape
    next_grid = grid.copy()

    # Step 1 (Parallel Phase): Each pedestrian independently chooses their best neighbor
    desired_moves = {}  # Maps target_cell -> list of source_cells wanting to move there
    pedestrians = np.argwhere(grid == 1)

    neighbors_offset = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    for r, c in pedestrians:
        best_cell = (r, c)
        best_score = floor_field[r, c]

        for dr, dc in neighbors_offset:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                # Can move into an exit (2) or currently empty space (0)
                if grid[nr, nc] in (0, 2):
                    # Add slight random noise to break ties naturally
                    score = floor_field[nr, nc] + np.random.uniform(0, 0.05)
                    if score < best_score:
                        best_score = score
                        best_cell = (nr, nc)

        if best_cell != (r, c):
            desired_moves.setdefault(best_cell, []).append((r, c))

    # Step 2 (Synchronous Conflict Resolution): Resolve collisions when multiple people pick the same cell
    evacuated_this_step = 0
    for target, candidates in desired_moves.items():
        tr, tc = target
        if grid[tr, tc] == 2:
            # Exits can absorb multiple people or one per door per tick; here one exits per door
            winner_r, winner_c = candidates[np.random.randint(len(candidates))]
            next_grid[winner_r, winner_c] = 0
            evacuated_this_step += 1
        elif next_grid[tr, tc] == 0:
            # Only one pedestrian can occupy an empty floor cell; pick one winner randomly
            winner_r, winner_c = candidates[np.random.randint(len(candidates))]
            next_grid[winner_r, winner_c] = 0
            next_grid[tr, tc] = 1

    return next_grid, evacuated_this_step


def render_ascii(grid):
    """Converts the numeric grid into readable ASCII characters for terminal display."""
    symbols = {-1: "#", 0: ".", 1: "P", 2: "E"}
    lines = []
    for row in grid:
        lines.append(" ".join(symbols[val] for val in row))
    return "\n".join(lines)


if __name__ == "__main__":
    room = initialize_room(rows=12, cols=20, num_people=30)
    floor_field = compute_floor_field(room)

    total_people = int(np.sum(room == 1))
    total_evacuated = 0
    time_step = 0

    print("=== Initial Room State (P = Person, # = Wall, E = Exit) ===")
    print(render_ascii(room))
    print(f"\nTotal Pedestrians: {total_people}\n")

    # Run synchronous cellular updates until everyone evacuates
    while np.sum(room == 1) > 0 and time_step < 200:
        time_step += 1
        room, evacuated = step_parallel_evacuation(room, floor_field)
        total_evacuated += evacuated

        if time_step % 5 == 0 or np.sum(room == 1) == 0:
            remaining = int(np.sum(room == 1))
            print(
                f"Step {time_step:2d} | Evacuated: {total_evacuated:2d}/{total_people} | Remaining: {remaining:2d}"
            )

    print("\n=== Final Room State ===")
    print(render_ascii(room))
    print(f"\nEvacuation completed in {time_step} parallel time steps.")
