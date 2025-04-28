import random
import heapq
import time
import pickle
import os
from math import sqrt

# ANSI color codes for terminal
class Colors:
    RESET = '\033[0m'
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'

class Node:
    def __init__(self, node_id, x, y, node_type="regular"):
        self.id = node_id
        self.x = x
        self.y = y
        self.type = node_type  # "regular", "start", "goal", or "barrier"
    
    def __lt__(self, other):
        # Required for priority queue operations
        return self.id < other.id
        
    def __str__(self):
        return f"Node {self.id} at ({self.x}, {self.y}) - {self.type}"
        
    def __repr__(self):
        return self.__str__()

class Maze:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.start_node = None
        self.goal_node = None
        self.barrier_nodes = []
        
    def setup_maze(self):
        # Create nodes with their coordinates using column-major ordering
        for x in range(6):
            for y in range(6):
                # Node ID = x * 6 + y (column-major ordering)
                node_id = x * 6 + y
                self.nodes[node_id] = Node(node_id, x, y)
        
        # Set up edges - each node connects to adjacent nodes (horizontal, vertical, diagonal)
        for node_id, node in self.nodes.items():
            self.edges[node_id] = []
            x, y = node.x, node.y
            
            # Check neighbors in all eight directions (horizontal, vertical, diagonal)
            potential_neighbors = [
                (x+1, y),    # right
                (x-1, y),    # left
                (x, y+1),    # down
                (x, y-1),    # up
                (x+1, y+1),  # down-right
                (x-1, y+1),  # down-left
                (x+1, y-1),  # up-right
                (x-1, y-1)   # up-left
            ]
            
            for nx, ny in potential_neighbors:
                if 0 <= nx < 6 and 0 <= ny < 6:
                    # Calculate neighbor ID using column-major ordering
                    neighbor_id = nx * 6 + ny
                    self.edges[node_id].append(neighbor_id)
    
    def select_special_nodes(self):
        # In column-major ordering:
        # - Nodes 0-11 are in the first two columns (x=0,1)
        # - Nodes 24-35 are in the last two columns (x=4,5)
        
        # Select start node (from nodes 0-11)
        start_candidates = []
        for x in range(2):  # First two columns (x=0,1)
            for y in range(6):
                start_candidates.append(x * 6 + y)
        start_id = random.choice(start_candidates)
        self.start_node = self.nodes[start_id]
        self.start_node.type = "start"
        
        # Select goal node (from nodes 24-35)
        goal_candidates = []
        for x in range(4, 6):  # Last two columns (x=4,5)
            for y in range(6):
                goal_candidates.append(x * 6 + y)
        goal_id = random.choice(goal_candidates)
        self.goal_node = self.nodes[goal_id]
        self.goal_node.type = "goal"
        
        # Select four barrier nodes from remaining nodes
        barrier_candidates = [i for i in range(36) if i != start_id and i != goal_id]
        barrier_ids = random.sample(barrier_candidates, 4)
        
        self.barrier_nodes = []  # Clear any existing barriers
        for barrier_id in barrier_ids:
            self.nodes[barrier_id].type = "barrier"
            self.barrier_nodes.append(self.nodes[barrier_id])
    
    def get_neighbors(self, node_id):
        """Get valid neighbors (excluding barriers) in increasing order of ID"""
        valid_neighbors = []
        for neighbor_id in self.edges[node_id]:
            if self.nodes[neighbor_id].type != "barrier":
                valid_neighbors.append(neighbor_id)
        return sorted(valid_neighbors)
    
    def calculate_edge_cost(self, node1_id, node2_id):
        """Calculate edge cost based on Euclidean distance between nodes"""
        node1 = self.nodes[node1_id]
        node2 = self.nodes[node2_id]
        
        # Calculate Euclidean distance
        dx = node1.x - node2.x
        dy = node1.y - node2.y
        
        # For diagonal movement (costs more than orthogonal)
        if dx != 0 and dy != 0:
            return sqrt(2)  # approx 1.414
        else:
            return 1.0  # Horizontal or vertical movement
    
    def print_maze(self, visited_nodes=None, path=None):
        """Print the maze in the terminal with colors and fixed-width cells"""
        visited_nodes = visited_nodes or []
        path = path or []
        
        # Helper function to format node IDs with consistent spacing
        def format_node_id(node_id, color=None):
            # All cells should have exactly 4 characters width
            if node_id < 10:
                formatted = f"  {node_id} "  # Two spaces, digit, space
            else:
                formatted = f" {node_id} "   # One space, two digits, space
            
            # Apply color if specified
            if color:
                return f"{color}{formatted}{Colors.RESET}"
            return formatted
        
        # Print column headers (x-coordinates)
        print("  ", end="")
        for x in range(6):
            print(f"  {x}  ", end="")
        print("\n")
        
        # Generate a lookup for nodes in the path to know if they're part of segment
        path_segments = {}
        if path and len(path) > 1:
            for i in range(len(path) - 1):
                # Current node and next node in path
                curr_node = self.nodes[path[i]]
                next_node = self.nodes[path[i+1]]
                
                # Determine direction to next node (for arrow display)
                dx = next_node.x - curr_node.x
                dy = next_node.y - curr_node.y
                
                # Store direction in path_segments
                path_segments[path[i]] = (dx, dy)
        
        # Print the maze with borders
        for y in range(6):
            # Print row header (y-coordinate)
            print(f"{y} ", end="")
            
            # Print top border of cells
            for x in range(6):
                print("+----", end="")
            print("+")
            
            # Print cell content
            print("  ", end="")
            for x in range(6):
                # Calculate node ID using column-major ordering
                node_id = x * 6 + y
                node = self.nodes[node_id]
                
                # Determine cell content based on node type and whether it's in path/visited
                if node_id in path:
                    if node.type == "start":
                        cell = f"{Colors.GREEN}{Colors.BOLD} S* {Colors.RESET}"
                    elif node.type == "goal":
                        cell = f"{Colors.RED}{Colors.BOLD} G* {Colors.RESET}"
                    else:
                        # Show directional arrow for path
                        if node_id in path_segments:
                            dx, dy = path_segments[node_id]
                            # Determine arrow symbol based on direction
                            if dx == 1 and dy == 0:
                                arrow = "→"  # right
                            elif dx == -1 and dy == 0:
                                arrow = "←"  # left
                            elif dx == 0 and dy == 1:
                                arrow = "↓"  # down
                            elif dx == 0 and dy == -1:
                                arrow = "↑"  # up
                            elif dx == 1 and dy == 1:
                                arrow = "↘"  # down-right
                            elif dx == -1 and dy == 1:
                                arrow = "↙"  # down-left
                            elif dx == 1 and dy == -1:
                                arrow = "↗"  # up-right
                            elif dx == -1 and dy == -1:
                                arrow = "↖"  # up-left
                            else:
                                arrow = "*"
                            
                            cell = f"{Colors.CYAN}{Colors.BOLD}  {arrow} {Colors.RESET}"
                        else:
                            # Last node in path (goal)
                            cell = f"{Colors.CYAN}{Colors.BOLD}  ● {Colors.RESET}"
                elif node.type == "start":
                    cell = f"{Colors.GREEN}{Colors.BOLD} S  {Colors.RESET}"
                elif node.type == "goal":
                    cell = f"{Colors.RED}{Colors.BOLD} G  {Colors.RESET}"
                elif node.type == "barrier":
                    cell = f"{Colors.BG_BLACK}{Colors.WHITE} ## {Colors.RESET}"
                elif node_id in visited_nodes:
                    cell = format_node_id(node_id, Colors.YELLOW)
                else:
                    cell = format_node_id(node_id)
                
                print(f"|{cell}", end="")
            print("|")
        
        # Print bottom border of the maze
        print("  ", end="")
        for x in range(6):
            print("+----", end="")
        print("+")
        
        # Print legend
        print("\nLegend:")
        print(f"{Colors.GREEN}{Colors.BOLD}S{Colors.RESET}   = Start node")
        print(f"{Colors.RED}{Colors.BOLD}G{Colors.RESET}   = Goal node")
        print(f"{Colors.BG_BLACK}{Colors.WHITE}##{Colors.RESET}  = Barrier")
        print(f"{Colors.CYAN}{Colors.BOLD}→↓←↑↗↘↙↖{Colors.RESET} = Path direction")
        print(f"{Colors.YELLOW}XX{Colors.RESET} = Visited node (not in path)")
        print()

def chebyshev_distance(node1, node2):
    """
    Calculate Chebyshev distance between two nodes:
    D(N,G) = max(|Nx-Gx|, |Ny-Gy|)
    """
    return max(abs(node1.x - node2.x), abs(node1.y - node2.y))

def save_maze(maze, filename="maze.pkl"):
    """Save the maze object to a file using pickle"""
    with open(filename, 'wb') as f:
        pickle.dump(maze, f)
    print(f"Maze saved to {filename}")

def load_maze(filename="maze.pkl"):
    """Load a maze object from a file"""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            maze = pickle.load(f)
        print(f"Maze loaded from {filename}")
        return maze
    else:
        print(f"No saved maze found at {filename}. Creating a new maze.")
        maze = Maze()
        maze.setup_maze()
        maze.select_special_nodes()
        save_maze(maze, filename)
        return maze

def main():
    # Create a new maze
    maze = Maze()
    maze.setup_maze()
    maze.select_special_nodes()
    
    print("New Maze Setup:")
    print(f"Start Node: {maze.start_node}")
    print(f"Goal Node: {maze.goal_node}")
    print("Barrier Nodes:")
    for barrier in maze.barrier_nodes:
        print(f"  {barrier}")
    
    # Display the maze in the terminal
    maze.print_maze()
    
    # Save the maze for later use by UCS and A* algorithms
    save_maze(maze)

if __name__ == "__main__":
    main()