import json
import tkinter as tk
from tkinter import Label, PhotoImage, simpledialog, Toplevel, Button, messagebox
import numpy as np
import time

# Component for reading data from sample.json
def read_river_data(file_path):
    with open(file_path, 'r') as file:
        river_data = json.load(file)
    return np.array(river_data)

# Function to display the river with obstacles
def display_river(canvas, river_matrix, path_boat, path_finish_flag):
    for row in range(river_matrix.shape[0]):
        for col in range(river_matrix.shape[1]):
            x1 = col * 30  # X-coordinate for the top-left corner
            y1 = row * 30  # Y-coordinate for the top-left corner
            x2 = x1 + 30   # X-coordinate for the bottom-right corner
            y2 = y1 + 30   # Y-coordinate for the bottom-right corner

            # Draw a cyan rectangle for the cell background (no border)
            canvas.create_rectangle(x1, y1, x2, y2, fill='cyan', outline='cyan')

            # Draw a rectangle for each cell (0: path, 1: obstacle)
            if row == 0 and col == 0:  # Boat starting position
                canvas.create_image(x1 + 15, y1 + 15, image=path_boat)
            elif row == 49 and col == 6:  # Finish position
                canvas.create_image(x1 + 15, y1 + 15, image=path_finish_flag)
            elif river_matrix[row, col] == 1:  # Obstacle
                canvas.create_rectangle(x1, y1, x2, y2, outline='', fill='black')  # Obstacle

# heuristic component for A* component
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# According to the coordinates of the current node, it returns
# the neighbors that are inside the boundaries
# of the matrix and are not obstacles.
def get_neighbors(river_matrix, current_node):
    neighbors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    for direction in directions:
        neighbor = (current_node[0] + direction[0], current_node[1] + direction[1])
        if 0 <= neighbor[0] < river_matrix.shape[0] and 0 <= neighbor[1] < river_matrix.shape[1]:
            if river_matrix[neighbor[0], neighbor[1]] != 1:  # Not an obstacle
                neighbors.append(neighbor)
    return neighbors

# Calculate energy consumption for a move
def calculate_energy(current, next_node):
    if next_node[0] < current[0]:  # Moving up
        return 1
    elif next_node[0] > current[0]:  # Moving down
        return 1
    elif next_node[1] < current[1]:  # Moving left
        return 1
    elif next_node[1] > current[1]:  # Moving right
        return 1
    elif next_node == current:  # Standing still
        return 1
    else:  # Moving backward
        return 2

# Breadth-First Search algorithm
def bfs(river_matrix, start, goal):
    queue = [(start, 100)]  # Include energy in the queue
    visited = set()
    parent = {start: None}
    nodes_explored = 0

    while queue:
        current, energy = queue.pop(0)
        nodes_explored += 1

        if current == goal:
            # Reconstruct the path
            path = []
            while current:
                path.append(current)
                current = parent[current]
            return path[::-1], nodes_explored

        visited.add(current)

        for neighbor in get_neighbors(river_matrix, current):
            if neighbor not in visited and all(neighbor != n for n, _ in queue):
                energy_cost = calculate_energy(current, neighbor)
                if energy - energy_cost >= 0:
                    queue.append((neighbor, energy - energy_cost))
                    parent[neighbor] = current

    return None, nodes_explored  # If no path is found

# Depth-First Search algorithm
def dfs(river_matrix, start, goal):
    stack = [(start, 100)]  # Include energy in the stack
    visited = set()
    parent = {start: None}
    nodes_explored = 0

    while stack:
        current, energy = stack.pop()
        nodes_explored += 1

        if current == goal:
            # Reconstruct the path
            path = []
            while current:
                path.append(current)
                current = parent[current]
            return path[::-1], nodes_explored

        if current not in visited:
            visited.add(current)

            for neighbor in get_neighbors(river_matrix, current):
                energy_cost = calculate_energy(current, neighbor)
                if neighbor not in visited and energy - energy_cost >= 0:
                    stack.append((neighbor, energy - energy_cost))
                    parent[neighbor] = current

    return None, nodes_explored  # If no path is found

# A* Search algorithm
def a_star(river_matrix, start, goal):
    open_set = [(start, 100)]  # Include energy in the open set
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    parent = {start: None}
    nodes_explored = 0

    while open_set:
        current, energy = min(open_set, key=lambda x: f_score.get(x[0], float('inf')))
        nodes_explored += 1

        if current == goal:
            # Reconstruct the path
            path = []
            while current:
                path.append(current)
                current = parent[current]
            return path[::-1], nodes_explored

        open_set.remove((current, energy))

        for neighbor in get_neighbors(river_matrix, current):
            tentative_g_score = g_score[current] + 1  # Assume all edges have weight 1
            energy_cost = calculate_energy(current, neighbor)

            if tentative_g_score < g_score.get(neighbor, float('inf')) and energy - energy_cost >= 0:
                parent[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                if all(neighbor != n for n, _ in open_set):
                    open_set.append((neighbor, energy - energy_cost))

    return None, nodes_explored  # If no path is found

# Function to animate the boat movement along the path
def animate_boat(river_matrix, path, algorithm, nodes_explored, execution_time):
    # Create a Tkinter window
    window = tk.Tk()
    window.title("Rowing Problem Animation")

    # Set up the canvas to draw the river matrix
    canvas_width = river_matrix.shape[1] * 30
    canvas_height = river_matrix.shape[0] * 30

    # Create a frame for scrollbar support
    frame = tk.Frame(window)
    frame.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(frame, width=canvas_width, height=canvas_height, scrollregion=(0, 0, canvas_width, canvas_height))
    hbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL, command=canvas.xview)
    vbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)

    canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
    hbar.pack(side=tk.BOTTOM, fill=tk.X)
    vbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Load images for boat and finish flag
    path_boat = PhotoImage(file="images/boat.png").subsample(10, 10)
    path_finish_flag = PhotoImage(file="images/flag-finish.png").subsample(10, 10)

    # Display the river
    display_river(canvas, river_matrix, path_boat, path_finish_flag)

    # Animate the boat along the path
    energy = 100
    for i, step in enumerate(path):
        x, y = step
        x1 = y * 30
        y1 = x * 30

        # Draw the boat at the current position
        boat = canvas.create_image(x1 + 15, y1 + 15, image=path_boat)

        # Update the canvas and pause for animation effect
        window.update()
        time.sleep(0.3)

        # Remove the boat from the previous position
        if i < len(path) - 1:
            next_step = path[i + 1]
            energy -= calculate_energy(step, next_step)
        canvas.delete(boat)

    # Keep the boat at the final position
    canvas.create_image(x1 + 15, y1 + 15, image=path_boat)

    # Show results
    show_results_window(algorithm, nodes_explored, execution_time, energy)
    window.mainloop()

# Function to display results after algorithm execution
def show_results_window(algorithm, nodes_explored, execution_time, remaining_energy):
    result_window = Toplevel()
    result_window.title("Algorithm Results")

    tk.Label(result_window, text=f"Algorithm: {algorithm}").pack()
    tk.Label(result_window, text=f"Nodes Explored: {nodes_explored}").pack()
    tk.Label(result_window, text=f"Execution Time: {execution_time:.2f} seconds").pack()
    tk.Label(result_window, text=f"Remaining Energy: {remaining_energy} units").pack()

    Button(result_window, text="Back to Algorithm Selection", command=result_window.destroy).pack()

# Function to choose algorithm and run it
def choose_algorithm(river_matrix, start, goal):
    while True:
        algorithm = simpledialog.askstring("Choose Algorithm", "Enter algorithm (BFS, DFS, A*):")
        if algorithm is None:
            break

        start_time = time.time()
        if algorithm.upper() == "BFS":
            path, nodes_explored = bfs(river_matrix, start, goal)
        elif algorithm.upper() == "DFS":
            path, nodes_explored = dfs(river_matrix, start, goal)
        elif algorithm.upper() == "A*":
            path, nodes_explored = a_star(river_matrix, start, goal)
        else:
            messagebox.showerror("Error", "Invalid algorithm. Please enter BFS, DFS, or A*.")
            continue

        execution_time = time.time() - start_time

        if path:
            print(f"{algorithm} Path:", path)
            animate_boat(river_matrix, path, algorithm, nodes_explored, execution_time)
        else:
            messagebox.showinfo("No Path", "No path found!")

# Example usage
file_path = 'sample.json'
start = (0, 0)
goal = (49, 6)
river_matrix = read_river_data(file_path)

choose_algorithm(river_matrix, start, goal)
