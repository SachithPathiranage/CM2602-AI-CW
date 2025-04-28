from maze_pathfinding import load_maze, chebyshev_distance, save_maze, Maze, Node
import heapq
import time

def astar_search(maze):
    """
    Implement A* Search algorithm with Chebyshev Distance heuristic.
    
    Returns:
        - visited_nodes: List of nodes visited in order
        - path: Final path from start to goal
        - time_to_goal: Time taken to find the goal (1 minute per node)
        - path_cost: Total cost of the final path
    """
    start_id = maze.start_node.id
    goal_id = maze.goal_node.id
    
    # Calculate initial heuristic
    start_heuristic = chebyshev_distance(maze.nodes[start_id], maze.nodes[goal_id])
    
    # Priority queue for frontier: (f_score, node_id, g_score, path)
    # f_score = g_score + heuristic
    frontier = [(start_heuristic, start_id, 0, [start_id])]
    
    explored = set()
    visited_nodes = []  # Keep track of nodes in order of visitation
    
    # Keep track of g_score for each node
    g_scores = {start_id: 0}
    
    while frontier:
        # Pop the node with lowest f_score
        f_score, current_id, g_score, path = heapq.heappop(frontier)
        
        # Skip if we've already explored this node with a better path
        if current_id in explored and g_score >= g_scores.get(current_id, float('inf')):
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
                "path_cost": g_score
            }
        
        # Mark as explored and update g_score
        explored.add(current_id)
        g_scores[current_id] = g_score
        
        # Get neighbors in increasing order of ID
        neighbors = maze.get_neighbors(current_id)
        
        # Explore neighbors
        for neighbor_id in neighbors:
            # Calculate tentative g_score for this neighbor
            edge_cost = maze.calculate_edge_cost(current_id, neighbor_id)
            tentative_g_score = g_score + edge_cost
            
            # Skip if we've found a better path to this neighbor already
            if neighbor_id in g_scores and tentative_g_score >= g_scores[neighbor_id]:
                continue
                
            # This is the best path so far to this neighbor
            new_path = path + [neighbor_id]
            
            # Calculate heuristic using Chebyshev Distance
            heuristic = chebyshev_distance(maze.nodes[neighbor_id], maze.nodes[goal_id])
            
            # Calculate f_score = g_score + heuristic
            new_f_score = tentative_g_score + heuristic
            
            # Add to frontier
            heapq.heappush(frontier, (new_f_score, neighbor_id, tentative_g_score, new_path))
    
    # If no path is found
    return {
        "visited_nodes": visited_nodes,
        "path": None,
        "time_to_goal": len(visited_nodes),
        "path_cost": float('inf')
    }

def run_astar():
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
    
    # Run A* algorithm
    print("\nRunning A* Search algorithm...")
    result = astar_search(maze)
    
    if result["path"]:
        print(f"\nA* Results:")
        print(f"Visited nodes list: {result['visited_nodes']}")
        print(f"Time to find goal: {result['time_to_goal']} minutes")
        print(f"Final path: {result['path']}")
        print(f"Path cost: {result['path_cost']:.2f}")
        
        # Visualize the path in the terminal
        print("\nA* Path Visualization:")
        maze.print_maze(result["visited_nodes"], result["path"])
    else:
        print("No path found!")
        maze.print_maze(result["visited_nodes"])
    
    return result

if __name__ == "__main__":
    run_astar()