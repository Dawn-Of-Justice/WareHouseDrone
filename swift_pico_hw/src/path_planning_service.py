#!/usr/bin/env python3

"""
# Team ID:          1284
# Theme:            WareHouse Drone
# Author List:      Salo E S, Govind S Sarath, Arjun G Ravi, Devanand A
# Filename:         WD_1284_path_planning_service.py
# Functions:        __init__, __hash__, __eq__, wait_for_random_points, goal_point_receiver, pixel_to_whycon, waypoint_callback, path_planning, _create_obstacle_grid, _in_collision, _edge_collides, _update_queue, _sample_uniform, _sample_ellipsoid, _process_best_edge, _distance, _is_potentially_better_path, _edge_value, plan, _extract_path, load_map, main
# Global variables: None
"""

import math
import random
import cv2
from typing import Set, Tuple, Optional, List
from dataclasses import dataclass

import numpy as np
from waypoint_navigation.srv import GetWaypoints
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Int32MultiArray
from rclpy.callback_groups import ReentrantCallbackGroup
import matplotlib.pyplot as plt

@dataclass
class Point:
    x: float
    y: float

    def __hash__(self):
        """
        Purpose:
        ---
        Give hash value of the values of the object

        Input Arguments:
        ---
        None

        Returns:
        ---
        `hash` :  [ int ]
            hash value of the object

        Example call:
        ---
        hash(Point(1.0, 2.0))
        """

        return hash((self.x, self.y))

def visualize_path(
    path: List[Point],
    obstacles: Set[Tuple[int, int]],
    start: Point,
    goal: Point,
    x_max: int = 1000,
    y_max: int = 1000
):
    """
    Visualize the planned path with obstacles, start, and goal points.
    
    Args:
        path: List of Points representing the planned path
        obstacles: Set of (x,y) tuples representing obstacle positions
        start: Starting Point
        goal: Goal Point
        x_max: Maximum x coordinate of the space
        y_max: Maximum y coordinate of the space
    """
    # Create figure and axis
    plt.figure(figsize=(12, 12))
    
    # Plot obstacles
    obstacle_x = [x for x, y in obstacles]
    obstacle_y = [y for x, y in obstacles]
    plt.scatter(obstacle_x, obstacle_y, c='black', s=1, alpha=0.5, label='Obstacles')
    
    # Plot path
    if path:
        path_x = [p.x for p in path]
        path_y = [p.y for p in path]
        plt.plot(path_x, path_y, 'b-', linewidth=2, label='Planned Path')
        
        # Add arrows to show direction
        for i in range(len(path)-1):
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
        plt.text(
            0.02, 0.98,
            f'Path Length: {total_distance:.1f} units\nWaypoints: {len(path)}',
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8),
            verticalalignment='top'
        )
    
    plt.tight_layout()
    plt.show()

class GraphNode:
    def __init__(self, point: Point):
        """
        Purpose:
        ---
        Initialises a Graph node

        Input Arguments:
        ---
        `point` : [ Point ]
            object that contains a point of x,y coordinates

        Returns:
        ---
        None

        Example call:
        ---
        GraphNode(Point(1.0, 2.0))
        """
        self.point = point
        self.parent: Optional["GraphNode"] = None
        self.cost = float("inf")

    def __eq__(self, other):
        """
        Purpose:
        ---
        compare two objects if they are equal

        Input Arguments:
        ---
        None

        Returns:
        ---
        `equal` :  [ bool ]
            value denoting whether the objects are equal or not.

        Example call:
        ---
        GraphNode(Point(1.0, 2.0)) == GraphNode(Point(2.0, 1.0))
        """
        if not isinstance(other, GraphNode):
            return False
        return self.point.x == other.point.x and self.point.y == other.point.y

    def __hash__(self):
        """
        Purpose:
        ---
        Give hash value of the values of the object

        Input Arguments:
        ---
        None

        Returns:
        ---
        `hash` :  [ int ]
            hash value of the object

        Example call:
        ---
        hash(GraphNode(Point(1.0, 2.0)))
        """
        return hash(self.point)


class WayPoints(Node):

    def __init__(self):
        """
        Purpose:
        ---
        Initialize the WayPoints node

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        WayPoints()
        """

        super().__init__("waypoints_service")
        self.callback_group = ReentrantCallbackGroup()
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        self.point_subscription = self.create_subscription(
            Int32MultiArray,
            "/random_points",
            self.goal_point_receiver,
            qos_profile,
            callback_group=self.callback_group,
        )
        self.goals = None
        self.current_waypoint = [500, 500]
        self.srv = self.create_service(
            GetWaypoints,
            "waypoints",
            self.waypoint_callback,
            callback_group=self.callback_group,
        )
        self.obstacles = load_map("2D_bit_map.png")
        self.count = 0
        self.path = []

        self.wait_for_random_points()

    def wait_for_random_points(self):
        """
        Purpose:
        ---
        Waits for the /random_points topic to receive data. It keeps the node
        in a waiting state until the goals are received from the /random_points topic.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        self.wait_for_random_points()
        """

        self.get_logger().info("Waiting for /random_points topic...")

        while rclpy.ok() and self.goals is None:
            self.get_logger().info("Waiting for initial points data...")
            rclpy.spin_once(self, timeout_sec=1.0)

        if self.goals is not None:
            self.get_logger().info("Received initial points data!")
        else:
            self.get_logger().error("Failed to receive points data!")

    def goal_point_receiver(self, msg):
        """
        Purpose:
        ---
        Receives the goal points from the /random_points topic and updates the goals attribute.

        Input Arguments:
        ---
        `msg` :  [ Int32MultiArray ]
            The message received from the /random_points topic containing the goal points data.

        Returns:
        ---
        None

        Example call:
        ---
        self.goal_point_receiver(msg)
        """

        try:
            self.goals = [
                Point(msg.data[0], msg.data[1]),
                Point(msg.data[2], msg.data[3]),
            ]
            self.get_logger().debug(f"Received new goals: {self.goals}")
        except Exception as e:
            self.get_logger().error(f"Error processing point data: {str(e)}")

    def pixel_to_whycon(self, imgx, imgy):
        """
        Purpose:
        ---
        Converts pixels from image coordinates to WhyCon coordinates.

        Input Arguments:
        ---
        `imgx` :  [ float ]
            x coordinate of pixel

        `imgy` :  [ float ]
            y coordinate of pixel

        Returns:
        ---
        `goal` :  [ array ]
            an array of x,y,z WhyCon coordinates

        Example call:
        ---
        self.pixel_to_whycon(100, 200)
        """

        goal_x = 0.02537 * imgx - 12.66
        goal_y = 0.02534 * imgy - 12.57
        goal_z = 27.0
        goal = [goal_x, goal_y, goal_z]
        return goal

    def waypoint_callback(self, request, response):
        """
        Purpose:
        ---
        Processes a waypoint request and generates a response giving the path between 2 points.

        Input Arguments:
        ---
        `request` :  [ object ]
            The request object containing the parameters for the waypoint request.

        `response` :  [ object ]
            The response object to be populated based on the request.

        Returns:
        ---
        `response` :  [ object ]
            The response object with the appropriate information based on the request.

        Example call:
        ---
        response = self.waypoint_callback(request, response)
        """

        if not request.get_waypoints:
            self.get_logger().info("Request rejected")
            return response

        # Wait for goals if they're not available
        if self.goals is None:
            self.get_logger().info("Waiting for goals to be available...")
            self.wait_for_random_points()

        if self.goals is None:
            self.get_logger().error("No goals available!")
            return response

        # Process waypoints
        while not self.path:
            start = Point(*self.current_waypoint)
            if self.count < len(self.goals):
                goal = self.goals[self.count]
                self.get_logger().info(f"Planning path from {start} to {goal}")
                self.path_planning(start, goal)

        self.count += 1

        # Prepare response
        response.waypoints.poses = [Pose() for _ in range(len(self.path))]
        for i, path_point in enumerate(self.path):
            point = self.pixel_to_whycon(path_point.x, path_point.y)
            response.waypoints.poses[i].position.x = point[0]
            response.waypoints.poses[i].position.y = point[1]
            response.waypoints.poses[i].position.z = point[2]

        self.get_logger().info("Waypoints ready")
        self.current_waypoint = [self.path[-1].x, self.path[-1].y]
        self.path = []

        return response

    def path_planning(self, start, goal):
        """
        Purpose:
        ---
        This function plans a path from the start point to the goal point using a path planning algotithm.
        We use a hu

        Input Arguments:
        ---
        `start` :  [ Point ]
            The starting point of the path. It should be an instance of the Point class.

        `goal` :  [ Point ]
            The goal point of the path. It should be an instance of the Point class.

        Returns:
        ---
        `samples` :  [ array ]
            A list of Point instances representing the sample points used for path planning.

        Example call:
        ---
        self.path_planning(start, goal)
        """

        samples = [
            Point(start.x, goal.y),
            Point(goal.x, start.y),
            Point((start.x + goal.x) // 2, (start.y + goal.y) // 2),
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
        ]

        planner = BITStarPlanner(start, goal, self.obstacles, samples)
        self.path = planner.plan()

        if self.path:
            print("Path found! Waypoints:")
            visualize_path(self.path, self.obstacles, start, goal)
            # planner.animate() # uncomment to visualise
        else:
            pass


class BITStarPlanner:
    def __init__(
        self, start: Point, goal: Point, obstacles: Set[Tuple[int, int]], samples
    ):
        """
        Purpose:
        ---
        Initializes the BIT* planner with the start and goal points, obstacles, and sample points.

        Input Arguments:
        ---
        `start` :  [ Point ]
            The starting point for the path planning.

        `goal` :  [ Point ]
            The goal point for the path planning.

        `obstacles` :  [ Set ]
            A set of coordinates representing the obstacles in the environment.

        `samples` :  [ List ]
            A list of sample points used for the BIT* algorithm.

        Returns:
        ---
        None

        Example call:
        ---
        BITStarPlanner(start, goal, obstacles, samples)
        """

        self.start = GraphNode(start)
        self.start.cost = 0
        self.goal = GraphNode(goal)
        self.obstacles = obstacles

        # Algorithm parameters
        self.samples_per_batch = 10
        self.max_iterations = 3000
        self.neighbor_radius = 200
        self.min_radius = 40

        # Cached obstacle lookup for faster collision checking
        self.obstacle_grid = self._create_obstacle_grid()

        # Search space bounds
        self.x_max = max(x for x, _ in obstacles)
        self.y_max = max(y for _, y in obstacles)

        # Initialize node sets
        self.vertices = {self.start}
        self.samples = {self.goal}
        self.queue = set()

        self.best_cost = float("inf")
        self.solution_found = False

        critical_samples = [
            s for s in samples[:10] if not self._in_collision(Point(s.x, s.y))
        ]
        for sample in critical_samples:
            self.samples.add(GraphNode(sample))

    def _create_obstacle_grid(self):
        """
        Purpose:
        ---
        Creates a grid representation of the obstacles in the search space.
        It is used to speed up collision checking by caching obstacle locations.

        Input Arguments:
        ---
        None

        Returns:
        ---
        `grid` :  [ dict ]
            A dictionary where the keys are tuples representing the coordinates of obstacles,
            and the values are set to True, indicating the presence of an obstacle at those coordinates.

        Example call:
        ---
        self._create_obstacle_grid()
        """

        grid = {}
        for x, y in self.obstacles:
            grid[(x, y)] = True
        return grid

    def _in_collision(self, point: Point) -> bool:
        """
        Purpose:
        ---
        Determines if a given point is in collision with any obstacles in the search space.

        Input Arguments:
        ---
        `point` :  [ Point ]
            A Point object representing the coordinates to be checked for collision.

        Returns:
        ---
        `in_collision` :  [ bool ]
            Returns True if the point is in collision with an obstacle, otherwise False.

        Example call:
        ---
        collision_status = self._in_collision(Point(2.5, 3.7))
        """

        x, y = int(round(point.x)), int(round(point.y))
        radius = 1

        # Quick boundary check
        if not (0 <= x <= self.x_max and 0 <= y <= self.y_max):
            return True

        # Check immediate neighborhood using grid
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if (x + dx, y + dy) in self.obstacle_grid:
                    return True
        return False

    def _edge_collides(self, start: Point, end: Point) -> bool:
        """
        Purpose:
        ---
        Checks if the edge between two points collides with any obstacles.

        Input Arguments:
        ---
        `start` :  [ Point ]
            A Point object representing the start coordinate of the edge.

        `end` :  [ Point ]
            A Point object representing the end coordinate of the edge.

        Returns:
        ---
        `in_collision` :  [ bool ]
            Returns True if the edge between the start and end points collides with an obstacle, otherwise False.

        Example call:
        ---
        self._edge_collides(Point(1.0, 2.0), Point(4.0, 5.0))
        """

        steps = max(int(self._distance(start, end) / 20), 5)  # Reduced resolution

        # Quick check of endpoints
        if self._in_collision(start) or self._in_collision(end):
            return True

        for i in range(1, steps):  # Skip endpoints
            t = i / steps
            x = start.x + t * (end.x - start.x)
            y = start.y + t * (end.y - start.y)
            if self._in_collision(Point(x, y)):
                return True
        return False

    def _update_queue(self):
        """
        Purpose:
        ---
        Clears and updates the queue with vertices within a reasonable distance.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        self._update_queue()
        """

        self.queue.clear()

        # Only consider vertices within a reasonable distance
        for vertex in self.vertices:
            vertex_to_goal = self._distance(vertex.point, self.goal.point)
            if vertex_to_goal > self.best_cost:
                continue

            for sample in self.samples:
                if self._distance(
                    vertex.point, sample.point
                ) <= self.neighbor_radius and self._is_potentially_better_path(
                    vertex, sample
                ):
                    self.queue.add((vertex, sample))

    def _sample_uniform(self):
        """
        Purpose:
        ---
        Generate a set of uniformly distributed sample points within the search space.

        Input Arguments:
        ---
        None

        Returns:
        ---
        `samples` :  [ set ]
            A set of uniformly distributed sample points.

        `attempts` :  [ int ]
            The number of attempts made to generate the samples.

        Example call:
        ---
        self._sample_uniform()
        """

        samples = set()
        attempts = 0
        max_attempts = self.samples_per_batch * 10

        while len(samples) < self.samples_per_batch and attempts < max_attempts:
            x = random.uniform(0, self.x_max)
            y = random.uniform(0, self.y_max)
            point = Point(x, y)
            new_node = GraphNode(point)

            if (
                not self._in_collision(point)
                and new_node not in samples
                and new_node not in self.samples
            ):
                samples.add(new_node)
            attempts += 1

        self.samples.update(samples)

    def _sample_ellipsoid(self):
        """
        Purpose:
        ---
        Samples points within an ellipsoid defined by the start and goal points
        and adds them to the set of samples if they are not in collision and not already sampled.

        Input Arguments:
        ---
        None

        Returns:
        ---
        `samples` :  [ set ]
            A set of sampled nodes within the ellipsoid.

        `attempts` :  [ int ]
            The number of attempts made to sample points.

        Example call:
        ---
        self._sample_ellipsoid()
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
        max_attempts = self.samples_per_batch * 10

        while len(samples) < self.samples_per_batch and attempts < max_attempts:
            theta = random.uniform(0, 2 * math.pi)
            r = math.sqrt(random.uniform(0, 1))

            x = center.x + x_radius * r * math.cos(theta)
            y = center.y + y_radius * r * math.sin(theta)
            point = Point(x, y)

            if (
                0 <= x <= self.x_max
                and 0 <= y <= self.y_max
                and not self._in_collision(point)
            ):
                samples.add(GraphNode(point))
            attempts += 1

        self.samples.update(samples)

    def _process_best_edge(self) -> bool:
        """
        Purpose:
        ---
        Processes the best edge from the queue of edges to be expanded in the graph search algorithm.

        Input Arguments:
        ---
        None

        Returns:
        ---
        `success` :  [ bool ]
            `True` if there are edges in the queue and it successfully processes the best edge, otherwise returns `False`.

        Example call:
        ---
        self._process_best_edge()
        """

        if not self.queue:
            return False

        best_edge = min(self.queue, key=lambda e: self._edge_value(e))
        self.queue.remove(best_edge)

        start_node, end_node = best_edge
        new_cost = start_node.cost + self._distance(start_node.point, end_node.point)

        if new_cost >= end_node.cost or self._edge_collides(
            start_node.point, end_node.point
        ):
            return False

        end_node.parent = start_node
        end_node.cost = new_cost

        if end_node in self.samples:
            self.samples.remove(end_node)
            self.vertices.add(end_node)

            for sample in self.samples:
                if self._is_potentially_better_path(end_node, sample):
                    self.queue.add((end_node, sample))

        return True

    @staticmethod
    def _distance(p1: Point, p2: Point) -> float:
        """
        Purpose:
        ---
        Ccalculates the Euclidean distance between two points.

        Input Arguments:
        ---
        `p1` :  [ Point ]
            The first point with x and y coordinates.

        `p2` :  [ Point ]
            The second point with x and y coordinates.

        Returns:
        ---
        `distance` :  [ float ]
            The Euclidean distance between the two points.

        Example call:
        ---
        self._distance(Point(1, 2), Point(4, 6))
        """

        return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)

    def _is_potentially_better_path(self, start: GraphNode, end: GraphNode) -> bool:
        """
        Purpose:
        ---
        Determine if the path from the start node to the end node is potentially better based on the distance.

        Input Arguments:
        ---
        `start` :  [ GraphNode ]
            The starting node of the path.

        `end` :  [ GraphNode ]
            The ending node of the path.

        Returns:
        ---
        `is_better` :  [ bool ]
            True if the path is potentially better, False otherwise.

        Example call:
        ---
        self._is_potentially_better_path(start_node, end_node)
        """

        if self._distance(start.point, end.point) > self.neighbor_radius:
            return False

        potential_cost = (
            start.cost
            + self._distance(start.point, end.point)
            + self._distance(end.point, self.goal.point)
        )

        return potential_cost < self.best_cost

    def _edge_value(self, edge: Tuple[GraphNode, GraphNode]) -> float:
        """
        Purpose:
        ---
        Calculate the value of an edge in the graph based on the cost of the start node.

        Input Arguments:
        ---
        `edge` :  [ Tuple ]
            A tuple containing the start and end nodes of the edge.

        Returns:
        ---
        `value` :  [ float ]
            The calculated value of the edge based on the start node's cost.

        Example call:
        ---
        self._edge_value((start_node, end_node))
        """

        start, end = edge
        return (
            start.cost
            + self._distance(start.point, end.point)
            + self._distance(end.point, self.goal.point)
        )

    def plan(self) -> List[Point]:
        """
        Purpose:
        ---
        Plan a path by iterating through a specified number of iterations and making improvements to the path.

        Input Arguments:
        ---
        None

        Returns:
        ---
        `path` :  [ List ]
            A list of points representing the planned path.

        Example call:
        ---
        planner.plan()
        """

        last_improvement = 0

        for iteration in range(self.max_iterations):
            if (
                iteration - last_improvement > 100
            ):  # Early termination if no improvement
                break

            if not self.queue:
                if len(self.samples) > 100:  # Limit sample size
                    break
                (
                    self._sample_uniform()
                    if not self.solution_found
                    else self._sample_ellipsoid()
                )
                self._update_queue()

            if not self._process_best_edge():
                continue

            if self.goal.cost < self.best_cost:
                self.best_cost = self.goal.cost
                self.solution_found = True
                last_improvement = iteration

                # Early exit if we find a good enough path
                if self.best_cost <= 1.2 * self._distance(
                    self.start.point, self.goal.point
                ):
                    break

        return self._extract_path()

    def _extract_path(self) -> List[Point]:
        """
        Purpose:
        ---
        Extracts the path from the start point to the goal point if a solution has been found.

        Input Arguments:
        ---
        None

        Returns:
        ---
        `path` :  [ List ]
            A list of points representing the path from the start point to the goal point.

        `empty_list` :  [ List ]
            An empty list if no solution has been found.

        Example call:
        ---
        self._extract_path()
        """

        if not self.solution_found:
            return []

        path = []
        current = self.goal
        while current is not None:
            path.append(current.point)
            current = current.parent

        return list(reversed(path))


def load_map(filepath: str) -> Set[Tuple[int, int]]:
    """
    Purpose:
    ---
    Loads a bit map image from the specified file path and returns a set of coordinates representing the obstacles.

    Input Arguments:
    ---
    `filepath` :  [ str ]
        The file path to the map image.

    Returns:
    ---
    `obstacles` :  [ Set ]
        A set of tuples representing the coordinates of the obstacles in the map.

    Example call:
    ---
    load_map('bit_map.png')
    """

    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {filepath}")

    obstacles = set()
    height, width = image.shape
    for y in range(height):
        for x in range(width):
            if image[y, x] == 0:  # Black pixels are obstacles
                obstacles.add((x, y))

    return obstacles


def main():
    """
    Purpose:
    ---
    Initialize the ros Node

    Input Arguments:
    ---
    None

    Returns:
    ---
    None

    Example call:
    ---
    main()
    """

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