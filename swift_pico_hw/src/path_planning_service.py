#!/usr/bin/env python3

"""
# Team ID:          1284
# Theme:            WareHouse Drone
# Author List:      Salo E S, Govind S Sarath, Arjun G Ravi, Devanand A
# Filename:         WD_1284_path_planning_service.py
# Functions:        Enhanced with improved path smoothing and collision avoidance
"""

import math
import time
import random
import cv2
import heapq
import numpy as np
from typing import Set, Tuple, Optional, List
from dataclasses import dataclass
from scipy.interpolate import splprep, splev
import joblib
from waypoint_navigation.srv import GetWaypoints
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from rclpy.callback_groups import ReentrantCallbackGroup

@dataclass
class Point:
    x: float
    y: float

    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

def save_path_visualization(
    path: List[Point],
    obstacles: Set[Tuple[int, int]],
    start: Point,
    goal: Point,
    filename: str = "/tmp/path_visualization.png",
    x_max: int = 1000,
    y_max: int = 1000
):
    """
    Save path visualization to a file instead of trying to display it.
    This avoids threading issues with matplotlib and GUI frameworks.
    
    Args:
        path: List of Points representing the planned path
        obstacles: Set of (x,y) tuples representing obstacle positions
        start: Starting Point
        goal: Goal Point
        filename: File to save the visualization to
        x_max: Maximum x coordinate of the space
        y_max: Maximum y coordinate of the space
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    import os
    
    # Create figure
    plt.figure(figsize=(12, 12))
    
    # Plot obstacles (sample if there are too many)
    obstacle_sample = list(obstacles)[:5000] if len(obstacles) > 5000 else obstacles
    obstacle_x = [x for x, y in obstacle_sample]
    obstacle_y = [y for _, y in obstacle_sample]
    plt.scatter(obstacle_x, obstacle_y, c='black', s=1, alpha=0.5, label='Obstacles')
    
    # Plot path
    if path:
        path_x = [p.x for p in path]
        path_y = [p.y for p in path]
        plt.plot(path_x, path_y, 'b-', linewidth=2, label='Planned Path')
        
        # Add arrows for direction (limit for performance)
        arrow_step = max(1, len(path)//20)
        for i in range(0, len(path)-1, arrow_step):
            dx = path[i+1].x - path[i].x
            dy = path[i+1].y - path[i].y
            plt.arrow(path[i].x, path[i].y, dx*0.2, dy*0.2,
                     head_width=15, head_length=20, fc='b', ec='b', alpha=0.5)
    
    # Plot start and goal points
    plt.plot(start.x, start.y, 'go', markersize=15, label='Start')
    plt.plot(goal.x, goal.y, 'ro', markersize=15, label='Goal')
    
    # Set plot properties
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)
    plt.grid(True, alpha=0.3)
    plt.title('Path Planning Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    
    # Add path statistics
    if path:
        total_distance = sum(
            np.sqrt((path[i+1].x - path[i].x)**2 + (path[i+1].y - path[i].y)**2)
            for i in range(len(path)-1)
        )
        
        # Calculate average turn angle (simplified)
        angle_samples = min(len(path)-2, 20)
        angle_step = max(1, (len(path)-2)//angle_samples)
        
        total_angle_change = 0
        angle_changes = []
        
        if len(path) > 2:
            for i in range(1, len(path)-1, angle_step):
                vec1 = (path[i].x - path[i-1].x, path[i].y - path[i-1].y)
                vec2 = (path[i+1].x - path[i].x, path[i+1].y - path[i].y)
                
                # Calculate angle between vectors
                dot_product = vec1[0]*vec2[0] + vec1[1]*vec2[1]
                mag1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
                mag2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
                
                # Avoid division by zero
                if mag1 > 0 and mag2 > 0:
                    cosine = dot_product / (mag1 * mag2)
                    # Clamp to avoid domain errors due to floating point
                    cosine = max(-1.0, min(1.0, cosine))
                    angle = math.acos(cosine) * 180 / math.pi
                    total_angle_change += angle
                    angle_changes.append(angle)
        
        avg_turn_angle = 0
        if angle_changes:
            avg_turn_angle = total_angle_change / len(angle_changes)
        
        plt.text(
            0.02, 0.98,
            f'Path Length: {total_distance:.1f} units\n'
            f'Waypoints: {len(path)}\n'
            f'Avg Turn Angle: {avg_turn_angle:.1f}°',
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8),
            verticalalignment='top'
        )
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    
    # Save to file
    plt.savefig(filename, dpi=150)
    plt.close()
    
    return filename

def interpolate_path(path, points_per_segment=5, min_distance=1.5):
    """
    Add uniformly spaced points between waypoints to create a smoother path,
    but only if the distance between waypoints is greater than min_distance
    
    Args:
        path: List of Points representing the planned path
        points_per_segment: Number of points to add between each pair of waypoints
        min_distance: Minimum distance between points to add interpolation (in units)
        
    Returns:
        List of Points with added intermediate waypoints
    """
    if len(path) < 2:
        return path
        
    smooth_path = [path[0]]  # Start with the first point
    
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i+1]
        
        # Calculate segment vector and distance
        dx = end.x - start.x
        dy = end.y - start.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Only add intermediate points if the distance is greater than min_distance
        if distance > min_distance:
            # Calculate number of points based on distance
            # More points for longer segments
            actual_points = max(2, min(int(distance / min_distance), points_per_segment))
            
            # Add intermediate points
            for j in range(1, actual_points):
                t = j / actual_points
                new_x = start.x + t * dx
                new_y = start.y + t * dy
                smooth_path.append(Point(new_x, new_y))
        
        # Add the endpoint (except for the last segment where we'll add it at the end)
        if i < len(path) - 2:
            smooth_path.append(end)
    
    # Add the final endpoint
    smooth_path.append(path[-1])
    
    return smooth_path

class QuadTree:
    def __init__(self, x, y, width, height, max_points=4, max_depth=6, depth=0):
        """
        Initialize a quadtree for efficient spatial queries
        
        Args:
            x, y: Bottom-left corner coordinates
            width, height: Dimensions of the quadtree
            max_points: Maximum number of points before subdivision
            max_depth: Maximum depth of the tree
            depth: Current depth of the tree
        """
        self.boundary = (x, y, width, height)
        self.max_points = max_points
        self.max_depth = max_depth
        self.depth = depth
        self.points = []
        self.children = None  # NW, NE, SW, SE
    
    def subdivide(self):
        """Create four children for this quadtree node"""
        x, y, width, height = self.boundary
        half_width = width / 2
        half_height = height / 2
        
        # Create children in clockwise order: NW, NE, SE, SW
        nw = QuadTree(x, y + half_height, half_width, half_height, 
                      self.max_points, self.max_depth, self.depth + 1)
        ne = QuadTree(x + half_width, y + half_height, half_width, half_height, 
                      self.max_points, self.max_depth, self.depth + 1)
        se = QuadTree(x + half_width, y, half_width, half_height, 
                      self.max_points, self.max_depth, self.depth + 1)
        sw = QuadTree(x, y, half_width, half_height, 
                      self.max_points, self.max_depth, self.depth + 1)
        
        self.children = [nw, ne, se, sw]
        
        # Redistribute points to children
        for point in self.points:
            px, py = point
            for child in self.children:
                if self._is_in_boundary(px, py, child.boundary):
                    child.insert(point)
                    break
        
        self.points = []  # Clear points after redistribution
    
    def insert(self, point):
        """Insert a point into the quadtree"""
        x, y = point
        
        # Check if point is in boundary
        if not self._is_in_boundary(x, y, self.boundary):
            return False
        
        # Insert point if we have space and no children
        if len(self.points) < self.max_points and self.children is None:
            self.points.append(point)
            return True
        
        # Subdivide if we don't have children yet and not at max depth
        if self.children is None and self.depth < self.max_depth:
            self.subdivide()
        
        # Try to insert into children if we have them
        if self.children is not None:
            for child in self.children:
                if child.insert(point):
                    return True
        
        # If we're at max depth, just add to current node
        if self.depth >= self.max_depth:
            self.points.append(point)
            return True
            
        return False
    
    def query_range(self, range_rect):
        """Find all points within the given range"""
        found_points = []
        
        # Check if range intersects this node
        if not self._intersects(range_rect, self.boundary):
            return found_points
        
        # Check points at this level
        rx, ry, rw, rh = range_rect
        for point in self.points:
            px, py = point
            if (rx <= px <= rx + rw and ry <= py <= ry + rh):
                found_points.append(point)
        
        # Recursively check children
        if self.children is not None:
            for child in self.children:
                found_points.extend(child.query_range(range_rect))
                
        return found_points
    
    def query_radius(self, x, y, radius):
        """Find all points within the given radius of (x,y)"""
        # Use a square bounding box for the radius
        range_rect = (x - radius, y - radius, radius * 2, radius * 2)
        
        # Get points in the square
        candidates = self.query_range(range_rect)
        
        # Filter by actual radius
        result = []
        radius_squared = radius * radius
        for point in candidates:
            px, py = point
            if (x - px) ** 2 + (y - py) ** 2 <= radius_squared:
                result.append(point)
                
        return result
    
    @staticmethod
    def _is_in_boundary(x, y, boundary):
        """Check if a point is within a boundary rectangle"""
        bx, by, bw, bh = boundary
        return (bx <= x <= bx + bw) and (by <= y <= by + bh)
    
    @staticmethod
    def _intersects(r1, r2):
        """Check if two rectangles intersect"""
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2
        
        return not (x1 + w1 < x2 or 
                   x1 > x2 + w2 or 
                   y1 + h1 < y2 or 
                   y1 > y2 + h2)

class GraphNode:
    def __init__(self, point: Point):
        self.point = point
        self.parent: Optional["GraphNode"] = None
        self.cost = float("inf")
        self.g_cost = float("inf")  # Cost from start
        self.h_cost = 0  # Heuristic cost to goal
        self.f_cost = float("inf")  # f = g + h
        self.neighbors = []  # Store neighbors for path connections
        self.clearance = 0  # Distance to nearest obstacle

    def __eq__(self, other):
        if not isinstance(other, GraphNode):
            return False
        return self.point.x == other.point.x and self.point.y == other.point.y

    def __hash__(self):
        return hash(self.point)
    
    def __lt__(self, other):
        """Comparison operator for priority queue ordering"""
        if not isinstance(other, GraphNode):
            return NotImplemented
        return self.f_cost < other.f_cost or (self.f_cost == other.f_cost and self.h_cost < other.h_cost)

class EnhancedPathSmoother:
    """Advanced path smoother that creates natural-looking, smooth paths"""
    
    def __init__(self, path: List[Point], obstacles: Set[Tuple[int, int]], 
                 obstacle_quadtree: QuadTree, min_clearance: float = 5.0):
        self.path = path.copy()  # Work on a copy
        self.obstacles = obstacles
        self.quadtree = obstacle_quadtree
        self.min_clearance = min_clearance
        
    def _distance(self, p1: Point, p2: Point) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
    
    def _calculate_clearance(self, point: Point) -> float:
        """Calculate the clearance (distance to nearest obstacle)"""
        x, y = int(round(point.x)), int(round(point.y))
        
        # Start with a small radius and expand until we find obstacles
        for radius in range(1, 100, 5):  # Check up to 100 units away
            if self.quadtree:
                nearby_obstacles = self.quadtree.query_radius(x, y, radius)
                if nearby_obstacles:
                    # Calculate exact minimum distance
                    min_dist = float('inf')
                    for ox, oy in nearby_obstacles:
                        dist = math.sqrt((x - ox)**2 + (y - oy)**2)
                        min_dist = min(min_dist, dist)
                    return min_dist
            else:
                # Direct checking if no quadtree
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if (x + dx, y + dy) in self.obstacles:
                            return math.sqrt(dx**2 + dy**2)
                
        # If no obstacles found within range, return a large value
        return 100.0
    
    def _check_path_clearance(self, path: List[Point]) -> bool:
        """Check if all points in the path have sufficient clearance from obstacles"""
        for point in path:
            if self._calculate_clearance(point) < self.min_clearance:
                return False
        return True
    
    def _generate_spline_path(self, points: List[Point], num_points: int) -> List[Point]:
        """Generate a smooth spline path through the given points"""
        if len(points) < 3:
            return points
        
        # Extract x and y coordinates
        x = [p.x for p in points]
        y = [p.y for p in points]
        
        # Fit a spline
        tck, u = splprep([x, y], s=0.0, k=min(3, len(points)-1))
        
        # Generate points along the spline
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        
        # Create path of Points
        return [Point(x_new[i], y_new[i]) for i in range(len(x_new))]
    
    def _edge_collides(self, start: Point, end: Point) -> bool:
        """Check if an edge between two points collides with obstacles"""
        distance = self._distance(start, end)
        steps = max(int(distance / 10), 5)
        
        for i in range(1, steps):  # Skip endpoints
            t = i / steps
            x = start.x + t * (end.x - start.x)
            y = start.y + t * (end.y - start.y)
            
            # Check if point has sufficient clearance
            pt = Point(x, y)
            if self._calculate_clearance(pt) < self.min_clearance:
                return True
        
        return False
    
    def _curvature_cost(self, path: List[Point]) -> float:
        """Calculate a cost related to the curvature (smoothness) of the path"""
        if len(path) < 3:
            return 0.0
        
        total_angle_change = 0.0
        
        for i in range(1, len(path) - 1):
            # Calculate vectors
            v1 = (path[i].x - path[i-1].x, path[i].y - path[i-1].y)
            v2 = (path[i+1].x - path[i].x, path[i+1].y - path[i].y)
            
            # Calculate angle between vectors
            dot_product = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            # Avoid division by zero
            if mag1 > 0 and mag2 > 0:
                cosine = dot_product / (mag1 * mag2)
                # Clamp cosine to avoid domain errors
                cosine = max(-1.0, min(1.0, cosine))
                angle = math.acos(cosine)
                total_angle_change += angle
        
        return total_angle_change
    
    def _smooth_with_splines(self) -> List[Point]:
        """Apply spline-based smoothing to create a natural-looking path"""
        if len(self.path) < 3:
            return self.path
        
        # Generate spline path with more points for smoothness
        num_points = len(self.path) * 3
        smooth_path = self._generate_spline_path(self.path, num_points)
        
        # Check if the smoothed path maintains clearance from obstacles
        if self._check_path_clearance(smooth_path):
            # Further check edges between points
            for i in range(len(smooth_path) - 1):
                if self._edge_collides(smooth_path[i], smooth_path[i+1]):
                    # Fall back to original path if edge collision detected
                    return self._reduce_waypoints()
            
            # Successful smooth path
            return smooth_path
        
        # Fall back to waypoint reduction if spline doesn't work
        return self._reduce_waypoints()
    
    def _reduce_waypoints(self) -> List[Point]:
        """Reduce waypoints while maintaining path clearance"""
        if len(self.path) <= 2:
            return self.path
        
        result = [self.path[0]]  # Start with first point
        i = 0
        
        while i < len(self.path) - 1:
            # Look ahead as far as possible
            j = len(self.path) - 1
            valid_jump_found = False
            
            while j > i + 1:  # Must skip at least one point
                # Check if direct path is clear
                if not self._edge_collides(self.path[i], self.path[j]):
                    # Consider adding intermediate point for smoother turns
                    if j - i > 5:  # If skipping many points
                        # Add a midpoint for smoother path
                        mid_idx = (i + j) // 2
                        mid_point = self.path[mid_idx]
                        
                        # Check if path including midpoint has good clearance
                        if (not self._edge_collides(self.path[i], mid_point) and 
                            not self._edge_collides(mid_point, self.path[j])):
                            result.append(mid_point)
                    
                    result.append(self.path[j])
                    i = j
                    valid_jump_found = True
                    break
                j -= 1
                
            # If no valid jump found, move to next point
            if not valid_jump_found:
                i += 1
                if i < len(self.path):
                    result.append(self.path[i])
        
        return result
    
    def _optimize_turning_angles(self, path: List[Point]) -> List[Point]:
        """Optimize turning angles to make the path smoother"""
        if len(path) < 3:
            return path
            
        result = [path[0]]  # Keep first point
        
        # Process middle points
        for i in range(1, len(path)-1):
            prev = path[i-1]
            current = path[i]
            next_pt = path[i+1]
            
            # Calculate vectors
            v1 = (current.x - prev.x, current.y - prev.y)
            v2 = (next_pt.x - current.x, next_pt.y - current.y)
            
            # Calculate angle
            dot_product = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            # Check for sharp turn
            if mag1 > 0 and mag2 > 0:
                cosine = dot_product / (mag1 * mag2)
                # Clamp cosine to avoid domain errors
                cosine = max(-1.0, min(1.0, cosine))
                angle = math.acos(cosine) * 180 / math.pi
                
                # If turn is too sharp, try to adjust
                if angle > 30:  # Threshold for sharp turn in degrees
                    # Try to find a better point near the current one
                    found_better_point = False
                    
                    # Try points in a small circle around the current point
                    for r in range(5, 20, 5):  # Try different radii
                        for theta in range(0, 360, 30):  # Try different angles
                            rad_theta = math.radians(theta)
                            new_x = current.x + r * math.cos(rad_theta)
                            new_y = current.y + r * math.sin(rad_theta)
                            new_point = Point(new_x, new_y)
                            
                            # Calculate new vectors
                            new_v1 = (new_point.x - prev.x, new_point.y - prev.y)
                            new_v2 = (next_pt.x - new_point.x, next_pt.y - new_point.y)
                            
                            # Calculate new angle
                            new_dot = new_v1[0]*new_v2[0] + new_v1[1]*new_v2[1]
                            new_mag1 = math.sqrt(new_v1[0]**2 + new_v1[1]**2)
                            new_mag2 = math.sqrt(new_v2[0]**2 + new_v2[1]**2)
                            
                            if new_mag1 > 0 and new_mag2 > 0:
                                new_cosine = new_dot / (new_mag1 * new_mag2)
                                new_cosine = max(-1.0, min(1.0, new_cosine))
                                new_angle = math.acos(new_cosine) * 180 / math.pi
                                
                                # Check if new angle is better and has sufficient clearance
                                if new_angle < angle and self._calculate_clearance(new_point) >= self.min_clearance:
                                    # Check if new edges are clear
                                    if (not self._edge_collides(prev, new_point) and 
                                        not self._edge_collides(new_point, next_pt)):
                                        current = new_point
                                        found_better_point = True
                                        break
                        
                        if found_better_point:
                            break
            
            result.append(current)
        
        result.append(path[-1])  # Keep last point
        return result
    
    def smooth(self) -> List[Point]:
        """Apply a sequence of smoothing operations to create a high-quality path"""
        # First reduce waypoints to create a simpler path
        reduced_path = self._reduce_waypoints()
        
        # Try spline smoothing if path is complex enough
        if len(reduced_path) >= 4:
            spline_path = self._smooth_with_splines()
            
            # Compare curvature cost to see if spline path is smoother
            spline_cost = self._curvature_cost(spline_path)
            reduced_cost = self._curvature_cost(reduced_path)
            
            if spline_cost < reduced_cost:
                # If spline path is smoother, use it
                return spline_path
        
        # Otherwise use waypoint reduction with angle optimization
        return self._optimize_turning_angles(reduced_path)

class ImprovedPathPlanner:
    def __init__(
        self, 
        start: Point, 
        goal: Point, 
        obstacles: Set[Tuple[int, int]],
        obstacle_quadtree: Optional[QuadTree] = None,
        max_iterations: int = 3000,
        min_clearance: float = 4.5
    ):
        """
        Enhanced path planner with improved collision avoidance and smoothness
        """
        self.start = GraphNode(start)
        self.start.cost = 0
        self.start.g_cost = 0
        self.start.h_cost = self._distance(start, goal)
        self.start.f_cost = self.start.h_cost
        
        self.goal = GraphNode(goal)
        self.obstacles = obstacles
        self.obstacle_quadtree = obstacle_quadtree
        self.min_clearance = min_clearance

        # Algorithm parameters
        self.samples_per_batch = 150  # More samples for better coverage
        self.max_iterations = max_iterations
        self.min_radius = 40
        
        # Adaptive neighbor radius parameters
        self.neighbor_radius = 200
        self.min_neighbor_radius = 50
        self.radius_decay_factor = 0.98
        
        # Search space bounds
        if obstacles:
            self.x_max = max(x for x, _ in obstacles)
            self.y_max = max(y for _, y in obstacles)
        else:
            self.x_max = 1000
            self.y_max = 1000

        # Initialize node sets
        self.vertices = {self.start}
        self.samples = {self.goal}
        self.queue = []  # Priority queue for edges
        
        self.best_cost = float("inf")
        self.solution_found = False
        
        # For early termination
        self.last_improvement = 0
        
        # Add critical samples in important areas
        critical_samples = [
            Point(50, 50),
            Point(472, 50),
            Point(480, 270),
            Point(300, 270),
            Point(300, 524),
            Point(300, 920),
            Point(50, 520),
            Point(50, 920),
            Point(320, 740),
            Point(400, 640),
            Point(955, 50),
            Point(630, 380),
            Point(560, 640),
            Point(600, 850),
            Point(828, 828),
            Point(840, 600),
            Point(950, 600),
            # Add more critical points to help navigate around obstacles
            Point(200, 200),
            Point(400, 400),
            Point(600, 600),
            Point(800, 800),
            Point(200, 800),
            Point(800, 200),
            Point(500, 300),
            Point(500, 700),
        ]
        
        # Add critical samples only if they have sufficient clearance
        for sample_point in critical_samples:
            if not self._in_collision(sample_point, self.min_clearance):
                sample_node = GraphNode(sample_point)
                sample_node.clearance = self._calculate_clearance(sample_point)
                self.samples.add(sample_node)
    
    def _calculate_clearance(self, point: Point) -> float:
        """Calculate the distance to the nearest obstacle"""
        x, y = int(round(point.x)), int(round(point.y))
        
        # Start with a small radius and expand until we find obstacles
        for radius in range(1, 100, 5):  # Check up to 100 units away
            if self.obstacle_quadtree:
                nearby_obstacles = self.obstacle_quadtree.query_radius(x, y, radius)
                if nearby_obstacles:
                    # Calculate exact minimum distance
                    min_dist = float('inf')
                    for ox, oy in nearby_obstacles:
                        dist = math.sqrt((x - ox)**2 + (y - oy)**2)
                        min_dist = min(min_dist, dist)
                    return min_dist
            else:
                # Direct checking if no quadtree
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if (x + dx, y + dy) in self.obstacles:
                            return math.sqrt(dx**2 + dy**2)
                
        # If no obstacles found within range, return a large value
        return 100.0
    
    def _in_collision(self, point: Point, min_clearance: float = None) -> bool:
        """
        Check if a point is in collision with obstacles
        A point is in collision if it's closer than min_clearance to any obstacle
        """
        if min_clearance is None:
            min_clearance = self.min_clearance
            
        x, y = int(round(point.x)), int(round(point.y))
        
        # Quick boundary check
        if not (0 <= x <= self.x_max and 0 <= y <= self.y_max):
            return True
        
        clearance = self._calculate_clearance(point)
        return clearance < min_clearance

    def _edge_collides(self, start: Point, end: Point) -> bool:
        """
        Check if the edge between two points collides with obstacles
        Uses adaptive resolution based on edge length
        """
        # Calculate distance and adaptive resolution
        distance = self._distance(start, end)
        steps = max(int(distance / (10 if distance > 100 else 5)), 10)

        # Quick check of endpoints
        if self._in_collision(start) or self._in_collision(end):
            return True

        # Check points along the edge
        for i in range(1, steps - 1):  # Skip endpoints as they've been checked
            t = i / steps
            x = start.x + t * (end.x - start.x)
            y = start.y + t * (end.y - start.y)
            
            # Check with minimum clearance
            if self._in_collision(Point(x, y)):
                return True
                
        return False

    def _adaptive_sampling(self):
        """
        Implement adaptive sampling strategy for better coverage and smoother paths
        """
        # If we've found a solution, focus on improving it
        if self.solution_found:
            # Mix between ellipsoid sampling (80%) and clearance-biased sampling (20%)
            if random.random() < 0.8:
                self._sample_ellipsoid()
            else:
                self._sample_with_clearance_bias()
        else:
            # For initial exploration, mix strategies
            r = random.random()
            if r < 0.5:  # 50% chance for obstacle-aware sampling
                self._sample_with_clearance_bias()
            elif r < 0.8:  # 30% chance for uniform sampling
                self._sample_uniform()
            else:  # 20% chance for targeted sampling in challenging areas
                self._sample_near_challenging_areas()
    
    def _sample_with_clearance_bias(self):
        """
        Sample points with a bias toward areas with good clearance from obstacles
        This helps create smoother paths that don't get too close to obstacles
        """
        samples = set()
        attempts = 0
        max_attempts = self.samples_per_batch * 15  # More attempts to find good samples
        
        while len(samples) < self.samples_per_batch and attempts < max_attempts:
            # Sample randomly in the space
            x = random.uniform(0, self.x_max)
            y = random.uniform(0, self.y_max)
            point = Point(x, y)
            
            # Check if the point is collision-free
            if not self._in_collision(point):
                # Calculate clearance from obstacles
                clearance = self._calculate_clearance(point)
                
                # Accept with higher probability if clearance is good
                acceptance_prob = min(1.0, clearance / 30.0)  # Scale with clearance
                
                if random.random() < acceptance_prob:
                    new_node = GraphNode(point)
                    new_node.clearance = clearance
                    if new_node not in samples and new_node not in self.samples:
                        samples.add(new_node)
            
            attempts += 1
        
        self.samples.update(samples)
    
    def _sample_near_challenging_areas(self):
        """
        Sample points near obstacles and narrow passages to improve exploration
        """
        samples = set()
        attempts = 0
        max_attempts = self.samples_per_batch * 10
        
        while len(samples) < self.samples_per_batch and attempts < max_attempts:
            # Choose a random vertex as a reference point
            if self.vertices and random.random() < 0.5:
                ref_node = random.choice(list(self.vertices))
                ref_point = ref_node.point
                
                # Sample near the reference point
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(10, 100)  # Sample within reasonable range
                x = ref_point.x + distance * math.cos(angle)
                y = ref_point.y + distance * math.sin(angle)
            else:
                # Sample randomly in the space
                x = random.uniform(0, self.x_max)
                y = random.uniform(0, self.y_max)
            
            point = Point(x, y)
            
            # Only accept point if it has sufficient clearance from obstacles
            if not self._in_collision(point):
                new_node = GraphNode(point)
                new_node.clearance = self._calculate_clearance(point)
                
                if new_node not in samples and new_node not in self.samples:
                    samples.add(new_node)
            
            attempts += 1
        
        self.samples.update(samples)

    def _sample_uniform(self):
        """
        Generate uniformly distributed samples with minimum clearance from obstacles
        """
        samples = set()
        attempts = 0
        max_attempts = self.samples_per_batch * 15

        while len(samples) < self.samples_per_batch and attempts < max_attempts:
            x = random.uniform(0, self.x_max)
            y = random.uniform(0, self.y_max)
            point = Point(x, y)
            
            # Only accept points with sufficient clearance
            if not self._in_collision(point):
                new_node = GraphNode(point)
                new_node.clearance = self._calculate_clearance(point)
                
                if new_node not in samples and new_node not in self.samples:
                    samples.add(new_node)
            
            attempts += 1

        self.samples.update(samples)

    def _sample_ellipsoid(self):
        """
        Sample points within an ellipsoid defined by start, goal, and best cost
        """
        center = Point(
            (self.start.point.x + self.goal.point.x) / 2,
            (self.start.point.y + self.goal.point.y) / 2,
        )

        c_min = self._distance(self.start.point, self.goal.point)
        x_radius = self.best_cost / 2
        y_radius = math.sqrt(max(0, self.best_cost**2 - c_min**2)) / 2

        samples = set()
        attempts = 0
        max_attempts = self.samples_per_batch * 15  # More attempts for quality samples

        while len(samples) < self.samples_per_batch and attempts < max_attempts:
            # Sample inside ellipse
            theta = random.uniform(0, 2 * math.pi)
            r = math.sqrt(random.uniform(0, 1))

            x = center.x + x_radius * r * math.cos(theta)
            y = center.y + y_radius * r * math.sin(theta)
            
            if not (0 <= x <= self.x_max and 0 <= y <= self.y_max):
                attempts += 1
                continue
                
            point = Point(x, y)

            # Only accept points with sufficient clearance
            if not self._in_collision(point):
                new_node = GraphNode(point)
                new_node.clearance = self._calculate_clearance(point)
                
                if new_node not in self.samples:
                    samples.add(new_node)
            
            attempts += 1

        self.samples.update(samples)

    def _update_queue(self):
        """
        Update priority queue with potentially useful edges,
        prioritizing edges that lead to smoother paths
        """
        self.queue = []  # Clear the queue

        # Find edges within reasonable distance that could lead to better paths
        for vertex in self.vertices:
            vertex_to_goal = self._distance(vertex.point, self.goal.point)
            
            # Skip vertices that can't possibly lead to better solutions
            if vertex_to_goal > self.best_cost - vertex.cost:
                continue

            # Find suitable neighbors based on search state
            neighbors = []
            if self.solution_found:
                # Use k-nearest approach when improving a solution
                k = min(15, len(self.samples))  # More neighbors for better connectivity
                if k > 0:
                    neighbors = sorted(
                        self.samples,
                        key=lambda s: self._distance(vertex.point, s.point)
                    )[:k]
            else:
                # Use radius-based approach when exploring
                for sample in self.samples:
                    if self._distance(vertex.point, sample.point) <= self.neighbor_radius:
                        neighbors.append(sample)
            
            # Add potentially useful edges to queue
            for sample in neighbors:
                if self._is_potentially_better_path(vertex, sample):
                    # Calculate smoothness factor
                    smoothness = self._evaluate_smoothness(vertex, sample)
                    
                    # Combine edge value with smoothness for prioritization
                    edge_value = self._edge_value((vertex, sample))
                    # Lower value = higher priority
                    priority = edge_value * (1.0 + 0.2 * smoothness)  # Penalize non-smooth edges
                    
                    heapq.heappush(self.queue, (priority, (vertex, sample)))
    
    def _evaluate_smoothness(self, vertex: GraphNode, sample: GraphNode) -> float:
        """
        Evaluate how smooth the path would be if we added this edge
        Lower values indicate smoother paths
        """
        # If vertex has no parent, can't evaluate smoothness
        if vertex.parent is None:
            return 0.0
            
        # Calculate vectors
        v1 = (vertex.point.x - vertex.parent.point.x, 
              vertex.point.y - vertex.parent.point.y)
        v2 = (sample.point.x - vertex.point.x, 
              sample.point.y - vertex.point.y)
        
        # Calculate the angle between vectors
        dot_product = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 > 0 and mag2 > 0:
            cosine = dot_product / (mag1 * mag2)
            # Clamp cosine to avoid domain errors
            cosine = max(-1.0, min(1.0, cosine))
            angle = math.acos(cosine)
            
            # Return angle as a measure of non-smoothness (0 = straight line, π = reversal)
            return angle / math.pi  # Normalize to [0, 1]
            
        return 0.5  # Default middle value if can't calculate
    
    def _process_best_edge(self) -> bool:
        """
        Process the best edge from the priority queue
        """
        if not self.queue:
            return False

        # Get the best edge from the queue
        _, best_edge = heapq.heappop(self.queue)
        start_node, end_node = best_edge
        
        # Calculate the new cost to end_node through start_node
        new_cost = start_node.cost + self._distance(start_node.point, end_node.point)

        # Skip if not an improvement or edge collides
        if new_cost >= end_node.cost or self._edge_collides(start_node.point, end_node.point):
            return False

        # Update node with better path
        end_node.parent = start_node
        end_node.cost = new_cost
        
        # If the end node was a sample, move it to vertices
        if end_node in self.samples:
            self.samples.remove(end_node)
            self.vertices.add(end_node)
            
            # Update best cost if we've reached the goal
            if end_node == self.goal:
                self.solution_found = True
                self.best_cost = new_cost
                self.last_improvement = self.iterations
                
                # Decay neighbor radius to focus search
                self.neighbor_radius = max(
                    self.min_neighbor_radius,
                    self.neighbor_radius * self.radius_decay_factor
                )
            
            # Add edges from this new vertex to nearby samples
            for sample in self.samples:
                if self._is_potentially_better_path(end_node, sample):
                    # Calculate edge priority with smoothness factor
                    smoothness = self._evaluate_smoothness(end_node, sample)
                    edge_value = self._edge_value((end_node, sample))
                    priority = edge_value * (1.0 + 0.2 * smoothness)
                    
                    heapq.heappush(self.queue, (priority, (end_node, sample)))

        return True

    @staticmethod
    def _distance(p1: Point, p2: Point) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)

    def _is_potentially_better_path(self, start: GraphNode, end: GraphNode) -> bool:
        """Check if a path through 'start' to 'end' could potentially be better"""
        # Check distance constraint
        if self._distance(start.point, end.point) > self.neighbor_radius:
            return False

        # Calculate potential cost of path
        potential_cost = (
            start.cost
            + self._distance(start.point, end.point)
            + self._distance(end.point, self.goal.point)
        )

        # Return true if this path is potentially better than current best
        return potential_cost < self.best_cost

    def _edge_value(self, edge: Tuple[GraphNode, GraphNode]) -> float:
        """
        Calculate edge priority value, accounting for both distance and clearance
        """
        start, end = edge
        
        # Calculate basic costs
        g_cost = start.cost + self._distance(start.point, end.point)
        h_cost = self._distance(end.point, self.goal.point)
        f_cost = g_cost + h_cost
        
        # Calculate clearance factor - prefer paths with better clearance
        clearance = end.clearance if hasattr(end, 'clearance') else self._calculate_clearance(end.point)
        clearance_factor = max(1.0, 5.0 / max(0.1, clearance))  # Higher for lower clearance
        
        # Return combined value (lower is better)
        return f_cost * clearance_factor

    def plan(self) -> List[Point]:
        """
        Run the path planning algorithm to find a smooth, collision-free path
        with fallback mechanisms if no path is found
        """
        # First attempt with standard clearance
        path = self._plan_with_params(self.min_clearance)
        
        # If no path found, try with reduced clearance
        if not path:
            print("No path found with standard clearance, trying with reduced clearance")
            reduced_clearance = max(2.0, self.min_clearance * 0.6)
            path = self._plan_with_params(reduced_clearance)
            
            # If still no path, try minimal clearance (last resort)
            if not path:
                print("Still no path, attempting with minimal clearance")
                path = self._plan_with_params(1.0)  # Minimal safe clearance
                
                # If all planning attempts fail, try direct connection if possible
                if not path and not self._edge_collides(self.start.point, self.goal.point):
                    print("Using direct path as last resort")
                    return [self.start.point, self.goal.point]
        
        return path
        
    def _plan_with_params(self, clearance: float) -> List[Point]:
        """
        Internal planning method with specified clearance parameter
        """
        # Save original clearance to restore later
        original_clearance = self.min_clearance
        self.min_clearance = clearance
        
        self.iterations = 0
        self.last_improvement = 0

        start_time = time.time()
        time_limit = 3.5  # Increased time limit for more thorough planning
        
        # Reset planning state
        self.queue = []
        self.vertices = {self.start}
        self.samples = {self.goal}
        self.best_cost = float("inf")
        self.solution_found = False
        
        while self.iterations < self.max_iterations:
            # Check time limit
            if time.time() - start_time > time_limit:
                print(f"Time limit reached after {self.iterations} iterations")
                break
            
            # Stop if no improvement for a while
            if self.solution_found and (self.iterations - self.last_improvement > 200):
                break

            # Update the queue if needed
            if not self.queue:
                # If we have too many samples, stop adding more
                if len(self.samples) > 1000:  # Increased from 700 for better coverage
                    break
                
                # Sample new points
                self._adaptive_sampling()
                
                # Update the queue with new potential edges
                self._update_queue()
            
            # Process the best edge
            if not self._process_best_edge():
                continue

            # Early exit if we find a good enough path
            if self.solution_found:
                direct_dist = self._distance(self.start.point, self.goal.point)
                if self.best_cost <= 1.2 * direct_dist:  # Relaxed criterion slightly
                    break
            
            self.iterations += 1

        # Get raw path
        raw_path = self._extract_path()
        
        # Restore original clearance
        self.min_clearance = original_clearance
        
        # If path found, apply enhanced smoothing
        if raw_path and len(raw_path) > 2:
            smoother = EnhancedPathSmoother(raw_path, self.obstacles, self.obstacle_quadtree, clearance)
            return smoother.smooth()
            
        return raw_path

    def _extract_path(self) -> List[Point]:
        """Extract the path from goal to start by following parent pointers"""
        if not self.solution_found:
            return []

        path = []
        current = self.goal
        
        while current is not None:
            path.append(current.point)
            current = current.parent

        return list(reversed(path))  # Reverse to get start-to-goal order

class RRTPlanner:
    """Rapidly-exploring Random Tree (RRT) path planner"""
    
    def __init__(
        self, 
        start: Point, 
        goal: Point, 
        obstacles: Set[Tuple[int, int]],
        obstacle_quadtree: Optional[QuadTree] = None,
        max_iterations: int = 5000,
        step_size: float = 20.0,
        goal_sample_rate: float = 0.3,
        min_clearance: float = 1.0
    ):
        """
        Initialize RRT planner
        
        Args:
            start: Starting point
            goal: Goal point
            obstacles: Set of obstacle coordinates (x,y)
            obstacle_quadtree: Optional quadtree for efficient obstacle lookups
            max_iterations: Maximum iterations to attempt path finding
            step_size: Maximum distance for each step in the tree
            goal_sample_rate: Probability of sampling the goal point directly
            min_clearance: Minimum clearance from obstacles
        """
        self.start = GraphNode(start)
        self.goal = GraphNode(goal)
        self.obstacles = obstacles
        self.obstacle_quadtree = obstacle_quadtree
        self.min_clearance = min_clearance
        
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        
        # Search space bounds
        if obstacles:
            self.x_max = max(x for x, _ in obstacles)
            self.y_max = max(y for _, y in obstacles)
        else:
            self.x_max = 1000
            self.y_max = 1000
            
        # Node tracking
        self.vertices = [self.start]
        self.goal_threshold = 20.0  # Distance to consider goal reached
    
    def _in_collision(self, point: Point) -> bool:
        """Check if a point is in collision with obstacles"""
        x, y = int(round(point.x)), int(round(point.y))
        radius = max(1, int(self.min_clearance))
        
        # Quick boundary check
        if not (0 <= x <= self.x_max and 0 <= y <= self.y_max):
            return True
        
        # Use quadtree for efficient collision checking
        if self.obstacle_quadtree:
            nearby_obstacles = self.obstacle_quadtree.query_radius(x, y, radius)
            return len(nearby_obstacles) > 0
        
        # Fallback to direct checking
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if (x + dx, y + dy) in self.obstacles:
                    return True
        return False
    
    def _edge_collides(self, start: Point, end: Point) -> bool:
        """Check if an edge between two points collides with obstacles"""
        distance = self._distance(start, end)
        steps = max(int(distance / 10), 5)
        
        for i in range(1, steps):  # Skip endpoints
            t = i / steps
            x = start.x + t * (end.x - start.x)
            y = start.y + t * (end.y - start.y)
            
            if self._in_collision(Point(x, y)):
                return True
        
        return False
    
    @staticmethod
    def _distance(p1: Point, p2: Point) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
    
    def _random_point(self) -> Point:
        """Generate a random point in the search space"""
        if random.random() < self.goal_sample_rate:
            return self.goal.point
        
        x = random.uniform(0, self.x_max)
        y = random.uniform(0, self.y_max)
        return Point(x, y)
    
    def _nearest_vertex(self, point: Point) -> GraphNode:
        """Find the nearest vertex in the tree to the given point"""
        return min(self.vertices, key=lambda v: self._distance(v.point, point))
    
    def _steer(self, from_point: Point, to_point: Point) -> Point:
        """Steer from from_point toward to_point with limited step size"""
        dist = self._distance(from_point, to_point)
        
        if dist <= self.step_size:
            return to_point
        
        # Calculate new point at step_size distance from from_point toward to_point
        ratio = self.step_size / dist
        new_x = from_point.x + ratio * (to_point.x - from_point.x)
        new_y = from_point.y + ratio * (to_point.y - from_point.y)
        
        return Point(new_x, new_y)
    
    def plan(self) -> List[Point]:
        """Run RRT algorithm to find a path"""
        start_time = time.time()
        time_limit = 2.0  # 2 seconds time limit
        
        # Track if we've reached the goal
        goal_node = None
        
        for i in range(self.max_iterations):
            # Check time limit
            if time.time() - start_time > time_limit:
                print(f"RRT time limit reached after {i} iterations")
                break
                
            # Sample random point
            random_point = self._random_point()
            
            # Find nearest vertex
            nearest = self._nearest_vertex(random_point)
            
            # Steer toward random point
            new_point = self._steer(nearest.point, random_point)
            
            # Skip if collision
            if self._in_collision(new_point) or self._edge_collides(nearest.point, new_point):
                continue
            
            # Create new node
            new_node = GraphNode(new_point)
            new_node.parent = nearest
            
            # Add to vertices
            self.vertices.append(new_node)
            
            # Check if goal reached
            if self._distance(new_point, self.goal.point) <= self.goal_threshold:
                goal_node = new_node
                break
            
            # Check direct connection to goal periodically
            if i % 20 == 0 and not self._edge_collides(new_point, self.goal.point) and \
               self._distance(new_point, self.goal.point) < self.step_size * 2:
                # Connect directly to goal
                goal_node = GraphNode(self.goal.point)
                goal_node.parent = new_node
                break
        
        # Extract path if goal was reached
        if goal_node:
            path = []
            current = goal_node
            
            # Add goal as final point
            path.append(self.goal.point)
            
            # Follow path back to start
            while current and current != self.start:
                path.append(current.point)
                current = current.parent
            
            path.append(self.start.point)
            
            return list(reversed(path))
        
        return []  # No path found

class WayPoints(Node):
    def __init__(self):
        super().__init__("waypoints_service")
        self.callback_group = ReentrantCallbackGroup()
        self.current_waypoint = [500, 500]
        self.package_whycon_location = {
            0: [500, 500], # Origin
            1: [224, 256],
            2: [690, 257],
            3: [161, 736],
            4: [843, 587],
            5: [554, 683],
            6: [835, 931]
        }
        
        self.goals = [
                Point(self.package_whycon_location[4][0], 
                      self.package_whycon_location[4][1]),
                Point(self.package_whycon_location[3][0], 
                      self.package_whycon_location[3][1]),
                Point(self.package_whycon_location[0][0], 
                      self.package_whycon_location[0][1]),  # Origin, used to come back
            ]
        
        self.obstacles = load_map("/home/salo/pico_ws2/2D_bit_map.png")
        self.obstacle_quadtree = self._create_obstacle_quadtree()
        
        self.count = 0
        self.path = []
        
        self.model_x = joblib.load("src/swift_pico_hw/src/linear_model_x.pkl")
        self.model_y = joblib.load("src/swift_pico_hw/src/linear_model_y.pkl")
        self.scalar = joblib.load("src/swift_pico_hw/src/scaler_inverse.pkl")
        
        self.srv = self.create_service(
            GetWaypoints,
            "waypoints",
            self.waypoint_callback,
            callback_group=self.callback_group,
        )
    
    def _create_obstacle_quadtree(self):
        """Create a quadtree data structure for efficient obstacle collision checking"""
        # Find bounds of space
        if not self.obstacles:
            return None
            
        max_x = max(x for x, _ in self.obstacles)
        max_y = max(y for _, y in self.obstacles)
        
        # Create quadtree
        qt = QuadTree(0, 0, max_x + 1, max_y + 1)
        
        # Insert obstacles
        for obstacle in self.obstacles:
            qt.insert(obstacle)
            
        return qt
    
    def pixel_to_whycon(self, imgx, imgy):
        points = np.array([[imgx, imgy]])
        points_scaled = self.scalar.transform(points)

        goal_x = self.model_x.predict(points_scaled)
        goal_y = self.model_y.predict(points_scaled)
        goal_z = 26.0
        goal = [goal_x, goal_y, goal_z]
        return goal

    def waypoint_callback(self, request, response):
        if not request.get_waypoints:
            self.get_logger().info("Request rejected")
            return response

        # Wait for goals if they're not available
        if self.goals is None:
            self.get_logger().info("Waiting for goals to be available...")

        if self.goals is None:
            self.get_logger().error("No goals available!")
            return response

        # Process waypoints
        while not self.path:
            start = Point(*self.current_waypoint)
            if self.count < len(self.goals):
                goal = self.goals[self.count]
                self.get_logger().info(f"Planning path from {start} to {goal}")
                
                # Plan new path - always calculate fresh path without caching
                self.path_planning(start, goal, visualize=True)
        
        self.count += 1
        
        # Prepare response - handle interpolated path
        response.waypoints.poses = [Pose() for _ in range(len(self.path))]
        
        # Use batch conversion for efficiency with larger number of points
        for i, path_point in enumerate(self.path):
            point = self.pixel_to_whycon(path_point.x, path_point.y)
            response.waypoints.poses[i].position.x = float(point[0])
            response.waypoints.poses[i].position.y = float(point[1])
            response.waypoints.poses[i].position.z = float(point[2])

        self.get_logger().info(f"Waypoints ready: {len(self.path)} points")
        self.current_waypoint = [self.path[-1].x, self.path[-1].y]
        self.path = []

        return response

    def path_planning(self, start, goal, visualize=False):
        """Plan a path from start to goal with enhanced smoothness and collision avoidance"""
        
        # First try improved path planner
        planner = ImprovedPathPlanner(start, goal, self.obstacles, self.obstacle_quadtree)
        self.path = planner.plan()

        # If still no path found (very rare with the fallback mechanisms)
        if not self.path:
            self.get_logger().warn("Primary planner failed, attempting RRT fallback...")
            
            # Try RRT* as fallback with minimal clearance
            from concurrent.futures import ThreadPoolExecutor
            
            # Create and run multiple RRT planners with different parameters in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Different settings for parallel attempts
                futures = [
                    executor.submit(self._try_rrt_planning, start, goal, 1.0, 5000, 20.0),  # Standard
                    executor.submit(self._try_rrt_planning, start, goal, 0.5, 7000, 15.0)  # Aggressive
                ]
                
                # Use the first successful result
                for future in futures:
                    result = future.result()
                    if result:
                        self.path = result
                        self.get_logger().info("Found path using fallback planner")
                        break

        if self.path:
            self.get_logger().info(f"Raw path found with {len(self.path)} waypoints")
            
            # Add interpolated points for smoother movement, but only where needed
            original_path_length = len(self.path)
            self.path = interpolate_path(self.path, points_per_segment=5, min_distance=1.5)
            self.get_logger().info(f"Path interpolated from {original_path_length} to {len(self.path)} waypoints")
            
            if visualize:
                self.get_logger().info("Saving path visualization to file...")
                try:
                    # Generate a timestamp-based filename
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    vis_file = f"/tmp/path_{timestamp}.png"
                    
                    # Save visualization to file
                    saved_file = save_path_visualization(
                        self.path, self.obstacles, start, goal, 
                        filename=vis_file
                    )
                    self.get_logger().info(f"Path visualization saved to: {saved_file}")
                    import subprocess
                    subprocess.Popen(['xdg-open', saved_file])  # Opens with default image viewer
                    
                except Exception as e:
                    self.get_logger().error(f"Visualization error: {str(e)}")
        else:
            self.get_logger().error("No path found with any method!")
                        
    def _try_rrt_planning(self, start, goal, min_clearance, max_iterations, step_size, visualize=False):
        """
        Try RRT planning with given parameters.
        
        Args:
            start (Point): Start point for the path
            goal (Point): Goal point for the path
            min_clearance (float): Minimum clearance from obstacles
            max_iterations (int): Maximum iterations for the RRT algorithm
            step_size (float): Step size for RRT exploration
            visualize (bool): Whether to save path visualization to a file
            
        Returns:
            List[Point]: The found path, or empty list if no path is found
        """
        self.get_logger().info(f"Trying RRT with clearance={min_clearance}, iterations={max_iterations}, step={step_size}")
        
        try:
            # Create RRT planner with specified parameters
            planner = RRTPlanner(
                start, goal, self.obstacles, self.obstacle_quadtree,
                max_iterations=max_iterations, 
                step_size=step_size,
                goal_sample_rate=0.3, 
                min_clearance=min_clearance
            )
            
            # Run planning and get the path
            path = planner.plan()
            
            if path:
                self.get_logger().info(f"RRT planning successful with {len(path)} waypoints")
                
                # Post-process the path to add minimal smoothing and interpolation
                
                # If path is very sparse, add minimal interpolation to improve control
                if len(path) < 3:
                    path = interpolate_path(path, points_per_segment=3, min_distance=1.5)
                
                # For longer paths, try to apply light smoothing if there are enough points
                elif len(path) > 5:
                    # Try to use the enhanced smoother with reduced parameters
                    try:
                        smoother = EnhancedPathSmoother(path, self.obstacles, self.obstacle_quadtree, min_clearance)
                        smoothed_path = smoother.smooth()
                        
                        # Only use smoothed path if it didn't reduce points too much
                        if len(smoothed_path) >= len(path) * 0.5:
                            path = smoothed_path
                            self.get_logger().info(f"Applied smoothing to RRT path, now has {len(path)} waypoints")
                    except Exception as e:
                        self.get_logger().warn(f"Smoothing failed, using original RRT path: {str(e)}")
                
                # Save visualization to file if requested
                if visualize:
                    try:
                        # Generate a timestamp-based filename for RRT path
                        import datetime
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        vis_file = f"/tmp/rrt_path_{timestamp}.png"
                        
                        # Save visualization to file
                        saved_file = save_path_visualization(
                            path, self.obstacles, start, goal, 
                            filename=vis_file
                        )
                        self.get_logger().info(f"RRT path visualization saved to: {saved_file}")
                    except Exception as e:
                        self.get_logger().error(f"RRT visualization error: {str(e)}")
                
                return path
            else:
                self.get_logger().warn("RRT planning failed to find a path")
                return []
                
        except Exception as e:
            self.get_logger().error(f"Error in RRT planning: {str(e)}")
            return []
   
def load_map(filepath: str) -> Set[Tuple[int, int]]:
    """Load map image and extract obstacle coordinates"""
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {filepath}")

    obstacles = set()
    height, width = image.shape
    
    # Process the image more efficiently - only add black pixels as obstacles
    for y in range(height):
        for x in range(width):
            if image[y, x] == 0:  # Black pixels are obstacles
                obstacles.add((x, y))

    return obstacles

def main():
    rclpy.init()
    waypoints = WayPoints()

    try:
        rclpy.spin(waypoints)
    except KeyboardInterrupt:
        waypoints.get_logger().info("KeyboardInterrupt, shutting down.\n")
    finally:
        waypoints.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()