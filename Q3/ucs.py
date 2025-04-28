from maze_pathfinding import load_maze, save_maze, Maze, Node
import heapq
import time

def uniform_cost_search(maze):
    """
    Implement Uniform Cost Search algorithm to find the shortest path
    from start node to goal node in the maze.
    
    Returns:
        - visited_nodes: List of nodes visited in order
        - path: Final path from start to goal
        - time_to_goal: Time taken to find the goal (1 minute per node)
        - path_cost: Total cost of the final path
    """
    start_id = maze.start_node.id
    goal_id = maze.goal_node.id
    
    # Priority queue for frontier: (cumulative_cost, node_id, path)
    frontier = [(0, start_id, [start_id])]
    explored = set()
    visited_nodes = []  # Keep track of nodes in order of visitation
    
    while frontier:
        # Pop the node with lowest cumulative cost
        cumulative_cost, current_id, path = heapq.heappop(frontier)
        
        # Skip if we've already explored this node
        if current_id in explored:
            continue
        
        # Add to visited nodes list
        visited_nodes.append(current_id)
        
        # Check if we've reached the goal
        if current_id == goal_id:
            time_to_goal = len(visited_nodes)  # 1 minute per node explored
            return {
                "visited_nodes": visited_nodes,
                "path": path,
                "time_to_goal": time_to_goal,
                "path_cost": cumulative_cost
            }
        
        # Mark as explored
        explored.add(current_id)
        
        # Get neighbors in increasing order of ID
        neighbors = maze.get_neighbors(current_id)
        
        # Explore neighbors
        for neighbor_id in neighbors:
            if neighbor_id not in explored:
                # Calculate the cost to move to this neighbor
                edge_cost = maze.calculate_edge_cost(current_id, neighbor_id)
                new_cost = cumulative_cost + edge_cost
                new_path = path + [neighbor_id]
                
                # Add to frontier with priority based on cumulative cost
                heapq.heappush(frontier, (new_cost, neighbor_id, new_path))
    
    # If no path is found
    return {
        "visited_nodes": visited_nodes,
        "path": None,
        "time_to_goal": len(visited_nodes),
        "path_cost": float('inf')
    }

def run_ucs():
    # Load the maze that was previously generated
    maze = load_maze()
    
    print("Loaded Maze Configuration:")
    print(f"Start Node: {maze.start_node}")
    print(f"Goal Node: {maze.goal_node}")
    print("Barrier Nodes:")
    for barrier in maze.barrier_nodes:
        print(f"  {barrier}")
    
    # Display the initial maze
    print("\nInitial Maze:")
    maze.print_maze()
    
    # Run UCS algorithm
    print("\nRunning Uniform Cost Search algorithm...")
    result = uniform_cost_search(maze)
    
    if result["path"]:
        print(f"\nUCS Results:")
        print(f"Visited nodes list: {result['visited_nodes']}")
        print(f"Time to find goal: {result['time_to_goal']} minutes")
        print(f"Final path: {result['path']}")
        print(f"Path cost: {result['path_cost']:.2f}")
        
        # Visualize the path in the terminal
        print("\nUCS Path Visualization:")
        maze.print_maze(result["visited_nodes"], result["path"])
    else:
        print("No path found!")
        maze.print_maze(result["visited_nodes"])
    
    return result

if __name__ == "__main__":
    run_ucs()