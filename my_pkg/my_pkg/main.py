import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan, Image

from math import atan2, sqrt, sin, pi
import heapq
import numpy as np

from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

# A* 노드 클래스
class NodeAStar:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f


# 메인 네비게이션 노드
class IntegratedNavigation(Node):
    def __init__(self):
        super().__init__('integrated_navigation')

        # 주행 파라미터
        self.lookahead_dist = 0.5
        self.linear_vel = 0.2
        self.stop_tolerance = 0.15

        #로봇 크기 & 안전거리
        self.robot_radius = 0.2
        self.safe_margin = 0.1

        # LiDAR
        self.base_obs_dist = 0.35
        self.speed_gain = 1.0
        self.front_dist = 99.9
        self.left_dist = 99.9
        self.right_dist = 99.9

        # 맵
        self.map_data = None
        self.map_resolution = 0.05
        self.map_origin = [0.0, 0.0]
        self.map_width = 0
        self.map_height = 0
        self.inflation_cells = 1

        # 로봇 상태 
        self.current_pose = None
        self.current_yaw = 0.0

        # 경로
        self.global_path = []
        self.path_index = 0

        # 상태 플래그
        self.avoiding = False
        self.emergency_stop = False

        # Publisher
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_path = self.create_publisher(Path, '/planned_path', 10)

        # Subscriber
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # YOLO
        self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.bridge = CvBridge()
        self.yolo_model = YOLO('/home/good/mj_ws/src/my_pkg/yolov8n.pt')

        cv2.namedWindow('YOLOv8 Phone Monitor', cv2.WINDOW_NORMAL)

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Integrated Navigation Started")

    # Map / Pose / Scan
    def map_callback(self, msg):
        self.map_resolution = msg.info.resolution
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_origin = [
            msg.info.origin.position.x,
            msg.info.origin.position.y
        ]
        self.map_data = np.array(msg.data).reshape((self.map_height, self.map_width))
        self.inflation_cells = int((self.robot_radius + self.safe_margin) / self.map_resolution)

    def pose_callback(self, msg):
        self.current_pose = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ]
        q = msg.pose.pose.orientation
        self.current_yaw = atan2(
            2.0 * (q.w*q.z + q.x*q.y),
            1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        )

    def scan_callback(self, msg):
        front = msg.ranges[0:40] + msg.ranges[-40:]
        left = msg.ranges[20:70]
        right = msg.ranges[-70:-20]

        self.front_dist = self.get_min_dist(front)
        self.left_dist = self.get_min_dist(left)
        self.right_dist = self.get_min_dist(right)

    def get_min_dist(self, ranges):
        valid = [r for r in ranges if 0.05 < r < 10.0]
        return min(valid) if valid else 99.9

    # Goal Callback
    def goal_callback(self, msg):
        if self.map_data is None or self.current_pose is None:
            self.get_logger().warn("Map or Pose not ready")
            return

        start = self.world_to_grid(self.current_pose)
        goal = self.world_to_grid([
            msg.pose.position.x,
            msg.pose.position.y
        ])

        # Goal 안전성 검사
        if not self.is_safe_cell(goal[0], goal[1]):
            self.get_logger().warn("Goal too close to obstacle → searching safe goal")
            safe_goal = self.find_nearest_safe_goal(goal)
            if safe_goal is None:
                self.get_logger().error("No safe goal found")
                return
            goal = safe_goal

        path = self.run_astar(start, goal)
        if path:
            self.global_path = [self.grid_to_world(p) for p in path]
            self.path_index = 0
            self.publish_path_viz()
            self.get_logger().info("Global path planned")
        else:
            self.get_logger().error("A* failed")

    def is_safe_cell(self, y, x):
        for dy in range(-self.inflation_cells, self.inflation_cells + 1):
            for dx in range(-self.inflation_cells, self.inflation_cells + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.map_height and 0 <= nx < self.map_width:
                    if self.map_data[ny][nx] != 0:
                        return False
        return True

    def find_nearest_safe_goal(self, goal):
        max_r = self.inflation_cells * 4
        for r in range(1, max_r + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    ny, nx = goal[0] + dy, goal[1] + dx
                    if 0 <= ny < self.map_height and 0 <= nx < self.map_width:
                        if self.is_safe_cell(ny, nx):
                            return (ny, nx)
        return None

    def run_astar(self, start, end):
        open_list = []
        heapq.heappush(open_list, NodeAStar(None, start))
        visited = set()

        moves = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]

        while open_list:
            current = heapq.heappop(open_list)
            if current.position in visited:
                continue
            visited.add(current.position)

            if current.position == end:
                path = []
                while current:
                    path.append(current.position)
                    current = current.parent
                return path[::-1]

            for dy, dx in moves:
                ny = current.position[0] + dy
                nx = current.position[1] + dx
                if not (0 <= ny < self.map_height and 0 <= nx < self.map_width):
                    continue
                if not self.is_safe_cell(ny, nx):
                    continue

                node = NodeAStar(current, (ny, nx))
                node.g = current.g + 1
                node.h = sqrt((ny-end[0])**2 + (nx-end[1])**2)
                node.f = node.g + node.h
                heapq.heappush(open_list, node)

        return None

    # YOLO 객체 감지(감지하면 정지)
    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        frame = cv2.resize(frame, (640, 480))

        results = self.yolo_model(frame, verbose=False)
        phone_detected = False

        for box in results[0].boxes or []:
            cls = int(box.cls.cpu().numpy()[0])
            if self.yolo_model.names[cls] == 'cell phone':
                phone_detected = True

        self.emergency_stop = phone_detected
        cv2.imshow('YOLOv8 Phone Monitor', frame)
        cv2.waitKey(1)

    # 제어 루프
    def control_loop(self):
        if self.emergency_stop:
            self.get_logger().warn("정지하세요. 폰이 감지되었습니다.")
            self.stop_robot()
            return

        if not self.global_path or self.current_pose is None:
            return

        if self.front_dist < self.dynamic_obs_threshold():
            self.avoid_obstacle()
            self.avoiding = True
            return

        if self.avoiding:
            self.recover_to_path()
            self.avoiding = False

        # Pure Pursuit
        target = self.global_path[self.path_index]
        alpha = atan2(target[1]-self.current_pose[1], target[0]-self.current_pose[0]) - self.current_yaw
        alpha = (alpha + pi) % (2*pi) - pi

        cmd = Twist()
        cmd.linear.x = self.linear_vel
        cmd.angular.z = max(min(self.linear_vel * (2*sin(alpha)) / self.lookahead_dist, 1.0), -1.0)
        self.pub_cmd.publish(cmd)

    def dynamic_obs_threshold(self):
        return self.base_obs_dist + self.linear_vel * self.speed_gain

    def avoid_obstacle(self):
        cmd = Twist()
        cmd.linear.x = 0.05
        cmd.angular.z = 0.8 if self.left_dist > self.right_dist else -0.8
        self.pub_cmd.publish(cmd)

    def recover_to_path(self):
        dists = [
            sqrt((p[0]-self.current_pose[0])**2 + (p[1]-self.current_pose[1])**2)
            for p in self.global_path
        ]
        self.path_index = int(np.argmin(dists))

    def stop_robot(self):
        self.pub_cmd.publish(Twist())

    def world_to_grid(self, world):
        return (
            int((world[1]-self.map_origin[1]) / self.map_resolution),
            int((world[0]-self.map_origin[0]) / self.map_resolution)
        )

    def grid_to_world(self, grid):
        return [
            grid[1]*self.map_resolution + self.map_origin[0],
            grid[0]*self.map_resolution + self.map_origin[1]
        ]

    def publish_path_viz(self):
        msg = Path()
        msg.header.frame_id = 'map'
        for p in self.global_path:
            ps = PoseStamped()
            ps.pose.position.x = p[0]
            ps.pose.position.y = p[1]
            msg.poses.append(ps)
        self.pub_path.publish(msg)


# 메인 함수
def main(args=None):
    rclpy.init(args=args)
    node = IntegratedNavigation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

