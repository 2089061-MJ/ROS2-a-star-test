import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan, Image

from math import atan2, sqrt, sin, pi
import heapq
import numpy as np

from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

class NodeAStar:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f

class IntegratedNavigation(Node):
    def __init__(self):
        super().__init__('integrated_navigation')

        # 파라미터 설정
        self.lookahead_dist = 0.4
        self.linear_vel = 0.15
        self.stop_tolerance = 0.15
        self.robot_radius = 0.2
        self.safe_margin = 0.05  # 마진을 살짝 줄여 YOLO 판단 시간을 확보
        self.base_obs_dist = 0.3  # LiDAR 감지 거리를 살짝 줄여 YOLO가 먼저 반응하게 함
        self.speed_gain = 1.0

        # 상태 변수
        self.front_dist = 99.9
        self.left_dist = 99.9
        self.right_dist = 99.9
        self.map_data = None
        self.map_resolution = 0.05
        self.map_origin = [0.0, 0.0]
        self.map_width = 0
        self.map_height = 0
        self.inflation_cells = 1
        self.current_pose = None
        self.current_yaw = 0.0
        self.global_path = []
        self.path_index = 0
        self.avoiding = False
        
        # YOLO 및 지연 대응 변수
        self.emergency_stop = False
        self.detection_count = 0  
        self.is_analyzing = False # 현재 YOLO 처리 중인지 여부

        # QoS 및 Pub/Sub 설정
        image_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, depth=1)
        lidar_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, depth=5)

        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_path = self.create_publisher(Path, '/planned_path', 10)

        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, lidar_qos)
        self.create_subscription(Image, '/image_raw', self.image_callback, image_qos)

        self.bridge = CvBridge()
        self.yolo_model = YOLO('/home/good/mj_ws/src/my_pkg/yolov8n.pt')
        cv2.namedWindow('YOLOv8 Phone Monitor', cv2.WINDOW_NORMAL)

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("통합 네비게이션 시작")

    def map_callback(self, msg):
        self.map_resolution = msg.info.resolution
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_origin = [msg.info.origin.position.x, msg.info.origin.position.y]
        self.map_data = np.array(msg.data).reshape((self.map_height, self.map_width))
        self.inflation_cells = int((self.robot_radius + self.safe_margin) / self.map_resolution)

    def pose_callback(self, msg):
        self.current_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        q = msg.pose.pose.orientation
        self.current_yaw = atan2(2.0 * (q.w*q.z + q.x*q.y), 1.0 - 2.0 * (q.y*q.y + q.z*q.z))

    def scan_callback(self, msg):
        # 전방 범위를 60도(+-30도)로 제한하여 측면 벽 오탐지 방지
        front = msg.ranges[0:30] + msg.ranges[-30:]
        left = msg.ranges[30:90]
        right = msg.ranges[270:330]
        self.front_dist = self.get_min_dist(front)
        self.left_dist = self.get_min_dist(left)
        self.right_dist = self.get_min_dist(right)

    def get_min_dist(self, ranges):
        valid = [r for r in ranges if 0.1 < r < 3.5]
        return min(valid) if valid else 99.9

    def goal_callback(self, msg):
        if self.map_data is None or self.current_pose is None:
            self.get_logger().warn("지도 또는 위치 정보 대기 중...")
            return
        start = self.world_to_grid(self.current_pose)
        goal = self.world_to_grid([msg.pose.position.x, msg.pose.position.y])
        path = self.run_astar(start, goal)
        if path:
            self.global_path = [self.grid_to_world(p) for p in path]
            self.path_index = 0
            self.publish_path_viz()
            self.get_logger().info("새 경로 생성 완료")

    def is_safe_cell(self, y, x):
        for dy in range(-self.inflation_cells, self.inflation_cells + 1):
            for dx in range(-self.inflation_cells, self.inflation_cells + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.map_height and 0 <= nx < self.map_width:
                    if self.map_data[ny][nx] != 0: return False
        return True

    def run_astar(self, start, end):
        open_list = []
        heapq.heappush(open_list, NodeAStar(None, start))
        visited = set()
        while open_list:
            current = heapq.heappop(open_list)
            if current.position in visited: continue
            visited.add(current.position)
            if current.position == end:
                path = []
                while current:
                    path.append(current.position); current = current.parent
                return path[::-1]
            for dy, dx in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
                ny, nx = current.position[0] + dy, current.position[1] + dx
                if 0 <= ny < self.map_height and 0 <= nx < self.map_width and self.is_safe_cell(ny, nx):
                    node = NodeAStar(current, (ny, nx))
                    node.g = current.g + 1
                    node.h = sqrt((ny-end[0])**2 + (nx-end[1])**2)
                    node.f = node.g + node.h
                    heapq.heappush(open_list, node)
        return None

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            # 성능을 위해 입력 이미지 크기 축소 고려 가능 (예: cv2.resize)
            results = self.yolo_model(frame, verbose=False, conf=0.4)
            
            phone_detected = False
            for box in results[0].boxes:
                if self.yolo_model.names[int(box.cls[0])] == 'cell phone':
                    phone_detected = True
                    b = box.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                    cv2.putText(frame, "DETECTING...", (b[0], b[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    break

            if phone_detected:
                self.detection_count = min(self.detection_count + 1, 10) # 최대 10까지만 누적
            else:
                self.detection_count = max(self.detection_count - 1, 0) # 서서히 감소 (노이즈 대비)
            
            self.emergency_stop = (self.detection_count >= 2)
            cv2.imshow('YOLOv8 Phone Monitor', frame)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"이미지 오류: {e}")

    def control_loop(self):
        # [핵심 수정] 1순위: YOLO 감지가 한 번이라도 되었다면 즉시 멈추고 제어권 독점
        if self.detection_count > 0:
            if self.emergency_stop:
                self.get_logger().warn("[YOLO 정지] 스마트폰 감지됨")
            else:
                self.get_logger().info("[분석 중] 객체 확인을 위해 일시 정지")
            self.stop_robot()
            return

        if not self.global_path or self.current_pose is None:
            return

        # 2순위: LiDAR 장애물 회피 (YOLO가 감지되지 않을 때만 실행)
        if self.front_dist < (self.base_obs_dist + self.linear_vel):
            self.get_logger().info("[LiDAR] 벽 감지됨: 회피 기동")
            self.avoid_obstacle()
            self.avoiding = True
            return

        # 3순위: 경로 복귀 및 추종
        if self.avoiding:
            self.recover_to_path()
            self.avoiding = False

        target = self.global_path[self.path_index]
        dx, dy = target[0]-self.current_pose[0], target[1]-self.current_pose[1]
        dist = sqrt(dx**2 + dy**2)

        if dist < self.lookahead_dist and self.path_index < len(self.global_path)-1:
            self.path_index += 1
        
        if dist < self.stop_tolerance and self.path_index == len(self.global_path)-1:
            self.get_logger().info("목적지 도착!")
            self.stop_robot()
            self.global_path = []
            return

        alpha = atan2(dy, dx) - self.current_yaw
        alpha = (alpha + pi) % (2*pi) - pi
        
        cmd = Twist()
        cmd.linear.x = self.linear_vel
        cmd.angular.z = max(min(self.linear_vel * (2*sin(alpha)) / self.lookahead_dist, 1.2), -1.2)
        self.pub_cmd.publish(cmd)

    def avoid_obstacle(self):
        cmd = Twist()
        cmd.linear.x = 0.0 # 전진 배제, 제자리 회전
        cmd.angular.z = 0.6 if self.left_dist > self.right_dist else -0.6
        self.pub_cmd.publish(cmd)

    def recover_to_path(self):
        dists = [sqrt((p[0]-self.current_pose[0])**2 + (p[1]-self.current_pose[1])**2) for p in self.global_path]
        self.path_index = int(np.argmin(dists))

    def stop_robot(self):
        self.pub_cmd.publish(Twist())

    def world_to_grid(self, world):
        return (int((world[1]-self.map_origin[1]) / self.map_resolution),
                int((world[0]-self.map_origin[0]) / self.map_resolution))

    def grid_to_world(self, grid):
        return [grid[1]*self.map_resolution + self.map_origin[0],
                grid[0]*self.map_resolution + self.map_origin[1]]

    def publish_path_viz(self):
        msg = Path()
        msg.header.frame_id = 'map'
        for p in self.global_path:
            ps = PoseStamped()
            ps.pose.position.x, ps.pose.position.y = p[0], p[1]
            msg.poses.append(ps)
        self.pub_path.publish(msg)

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
