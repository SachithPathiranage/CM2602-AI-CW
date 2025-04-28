from maze_pathfinding import load_maze, save_maze, Maze, Node
from ucs import uniform_cost_search
from astar import astar_search
import statistics

def calculate_statistics(results):
    """Calculate mean and variance for time and path length"""
    times = [r['time_to_goal'] for r in results if r['path'] is not None]
    path_lengths = [len(r['path']) for r in results if r['path'] is not None]
    path_costs = [r['path_cost'] for r in results if r['path'] is not None]
    
    stats = {
        'time': {
            'mean': statistics.mean(times) if times else 'N/A',
            'variance': statistics.variance(times) if len(times) > 1 else 'N/A'
        },
        'path_length': {
            'mean': statistics.mean(path_lengths) if path_lengths else 'N/A',
            'variance': statistics.variance(path_lengths) if len(path_lengths) > 1 else 'N/A'
        },
        'path_cost': {
            'mean': statistics.mean(path_costs) if path_costs else 'N/A',
            'variance': statistics.variance(path_costs) if len(path_costs) > 1 else 'N/A'
        }
    }
    
    return stats

def compare_algorithms(num_mazes=3):
    """Run UCS and A* on multiple random mazes and compare results"""
    ucs_results = []
    astar_results = []
    
    for i in range(num_mazes):
        print(f"\n{'='*50}")
        print(f"Running Maze {i+1} of {num_mazes}")
        print(f"{'='*50}")
        
        # Create a new maze for each iteration
        maze = Maze()
        maze.setup_maze()
        maze.select_special_nodes()
        
        # Save this maze for potential later use
        save_maze(maze)
        
        print(f"Maze {i+1} Setup:")
        print(f"Start Node: {maze.start_node}")
        print(f"Goal Node: {maze.goal_node}")
        print("Barrier Nodes:")
        for barrier in maze.barrier_nodes:
            print(f"  {barrier}")
        
        # Display the maze
        print(f"\nMaze {i+1} Visualization:")
        maze.print_maze()
        
        # Run UCS
        print(f"\nRunning UCS on Maze {i+1}...")
        ucs_result = uniform_cost_search(maze)
        ucs_results.append(ucs_result)
        
        if ucs_result["path"]:
            print(f"UCS Results (Maze {i+1}):")
            print(f"Visited nodes: {len(ucs_result['visited_nodes'])} nodes")
            print(f"Time to find goal: {ucs_result['time_to_goal']} minutes")
            print(f"Path length: {len(ucs_result['path'])} nodes")
            print(f"Path cost: {ucs_result['path_cost']:.2f}")
            
            # Display the path
            print(f"\nUCS Path for Maze {i+1}:")
            maze.print_maze(ucs_result["visited_nodes"], ucs_result["path"])
        else:
            print(f"UCS found no path for Maze {i+1}")
        
        # Run A*
        print(f"\nRunning A* on Maze {i+1}...")
        astar_result = astar_search(maze)
        astar_results.append(astar_result)
        
        if astar_result["path"]:
            print(f"A* Results (Maze {i+1}):")
            print(f"Visited nodes: {len(astar_result['visited_nodes'])} nodes")
            print(f"Time to find goal: {astar_result['time_to_goal']} minutes")
            print(f"Path length: {len(astar_result['path'])} nodes")
            print(f"Path cost: {astar_result['path_cost']:.2f}")
            
            # Display the path
            print(f"\nA* Path for Maze {i+1}:")
            maze.print_maze(astar_result["visited_nodes"], astar_result["path"])
        else:
            print(f"A* found no path for Maze {i+1}")
    
    # Calculate statistics
    ucs_stats = calculate_statistics(ucs_results)
    astar_stats = calculate_statistics(astar_results)
    
    # Print summary
    print("\n" + "="*50)
    print("Summary Statistics:")
    print("="*50)
    
    print("\nUCS Statistics:")
    print(f"Time to Goal (minutes) - Mean: {ucs_stats['time']['mean']}, Variance: {ucs_stats['time']['variance']}")
    print(f"Path Length (nodes) - Mean: {ucs_stats['path_length']['mean']}, Variance: {ucs_stats['path_length']['variance']}")
    print(f"Path Cost - Mean: {ucs_stats['path_cost']['mean']:.2f}, Variance: {ucs_stats['path_cost']['variance']}")
    
    print("\nA* Statistics:")
    print(f"Time to Goal (minutes) - Mean: {astar_stats['time']['mean']}, Variance: {astar_stats['time']['variance']}")
    print(f"Path Length (nodes) - Mean: {astar_stats['path_length']['mean']}, Variance: {astar_stats['path_length']['variance']}")
    print(f"Path Cost - Mean: {astar_stats['path_cost']['mean']:.2f}, Variance: {astar_stats['path_cost']['variance']}")
    
    # Analysis
    print("\n" + "="*50)
    print("Algorithm Analysis:")
    print("="*50)
    
    # Completeness analysis
    ucs_complete = all(r['path'] is not None for r in ucs_results)
    astar_complete = all(r['path'] is not None for r in astar_results)
    
    print("\nCompleteness:")
    print(f"UCS: {'Complete' if ucs_complete else 'Incomplete'}")
    print(f"A*: {'Complete' if astar_complete else 'Incomplete'}")
    
    # Optimality analysis
    print("\nOptimality:")
    for i in range(num_mazes):
        if ucs_results[i]['path'] and astar_results[i]['path']:
            ucs_cost = ucs_results[i]['path_cost']
            astar_cost = astar_results[i]['path_cost']
            
            if abs(ucs_cost - astar_cost) < 0.0001:  # Account for floating point comparison
                print(f"Maze {i+1}: Both algorithms found optimal path with cost {ucs_cost:.2f}")
            else:
                print(f"Maze {i+1}: UCS path cost: {ucs_cost:.2f}, A* path cost: {astar_cost:.2f}")
        else:
            print(f"Maze {i+1}: Cannot compare optimality (path not found by one or both algorithms)")
    
    # Time complexity analysis
    print("\nTime Complexity (nodes expanded):")
    for i in range(num_mazes):
        ucs_nodes = len(ucs_results[i]['visited_nodes'])
        astar_nodes = len(astar_results[i]['visited_nodes'])
        
        print(f"Maze {i+1}: UCS expanded {ucs_nodes} nodes, A* expanded {astar_nodes} nodes")
        if astar_nodes < ucs_nodes:
            print(f"  A* was more efficient by {ucs_nodes - astar_nodes} nodes ({(ucs_nodes - astar_nodes) / ucs_nodes * 100:.1f}%)")
        elif ucs_nodes < astar_nodes:
            print(f"  UCS was more efficient by {astar_nodes - ucs_nodes} nodes ({(astar_nodes - ucs_nodes) / astar_nodes * 100:.1f}%)")
        else:
            print("  Both algorithms expanded the same number of nodes")

if __name__ == "__main__":
    compare_algorithms(3)