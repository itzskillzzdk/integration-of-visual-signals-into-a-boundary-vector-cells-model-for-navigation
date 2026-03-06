import rclpy
from rclpy.node import Node
import cv2
import numpy as np
# from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
import argparse
import pygame
import os
import csv


# Paramètres
visual_system_fps = 15  # Exemple : 15 FPS
pygame_screenSize = 200, 100

class VisualSystem(Node):
    def __init__(self):
        super().__init__("VisualSystem")

        # Publishers
        self.publisher_keypoints = self.create_publisher(Float64MultiArray, "/visualSystem/keypoints", 10)
        self.publisher_heights = self.create_publisher(Float64MultiArray, "/visualSystem/heights", 10)
        self.publisher_fps = self.create_publisher(Float64, "/visualSystem/fps", 10)
        self.publisher_delta_yx = self.create_publisher(Float64MultiArray, "/visualSystem/delta_yx", 10)

    

        # Subscriptions
        self.subscription_camera = self.create_subscription(Image, "/vehicle/camera1", self.callback_camera1, 10)
        self.publisher_delta_heights = self.create_publisher(Float64MultiArray, "/visualSystem/delta_heights", 10)
        self.publisher_redundant = self.create_publisher(Float64MultiArray, "/visualSystem/redundant_keypoints", 10)
        self.publisher_redundant_prev = self.create_publisher(Float64MultiArray, "/visualSystem/redundant_keypoints_previous", 10)
        self.publisher_distance_features = self.create_publisher(Float64MultiArray, "/visualSystem/distance_features", 10)


        self.global_camera1 = None
        self.global_camera1_flag = False

        # self.bridge = CvBridge()
        self.clock = pygame.time.Clock()
        self.last_clock_fps = 0
        self.flag_loop = True
        self.frame_counter = 0
        self.prev_gray = None
        self.prev_keypoints = None
        self.csv_file_path = "delta_heights_log.csv"
        self.csv_initialized = False

        # --- Debug features logging ---
        self.declare_parameter('log_features', False)   # active/désactive logs
        self.declare_parameter('log_every', 3)        # log 1 fois / N frames traitées
        self.log_every = int(self.get_parameter('log_every').get_parameter_value().integer_value or 10)



    def callback_camera1(self, data):
        self.global_camera1 = data
        self.global_camera1_stamp = data.header.stamp  
        self.global_camera1_flag = True

    def run(self):
        pygame.init()
        args = self.parse_arguments()

        if args.debug:
            pygame.display.set_caption("Debug Mode")
            pygame_screen = pygame.display.set_mode(pygame_screenSize)
            font = pygame.font.SysFont(None, 24)

        while self.flag_loop:
            self.clock.tick_busy_loop(visual_system_fps)
            rclpy.spin_once(self, timeout_sec=0.01)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.flag_loop = False

            if self.global_camera1_flag:
                self.frame_counter += 1
                if self.frame_counter % 3 != 0:
                    self.global_camera1_flag = False
                    continue
                self.process_data()
                self.global_camera1_flag = False

            current_clock_fps = round(self.clock.get_fps(), 2)
            if self.last_clock_fps != current_clock_fps:
                buffer_vector = Float64()
                buffer_vector.data = current_clock_fps
                self.publisher_fps.publish(buffer_vector)
                self.last_clock_fps = current_clock_fps

            if args.debug:
                pygame_screen.fill((0, 0, 0))
                value = self.clock.get_fps()
                img = font.render(f"FPS: {value:.2f}", True, (255, 0, 0))
                pygame_screen.blit(img, (1, 0))
                pygame.display.update()

        pygame.quit()

    def encode_height(self, keypoints, global_image_size):
        if keypoints.shape[0] == 0:
            return np.array([])

        poi_y = keypoints[:, 1] / global_image_size[0]
        return poi_y
    
    def write_deltas_to_csv(self, delta_heights):
        try:
            write_header = not self.csv_initialized and not os.path.exists(self.csv_file_path)

            with open(self.csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                if write_header:
                    writer.writerow([f"delta_{i}" for i in range(len(delta_heights))])
                    self.csv_initialized = True
                writer.writerow(delta_heights)
        except Exception as e:
            self.get_logger().error(f"Error writing to CSV: {e}")


    def rbf_distance_features(self, d_est, centers, sigma=0.5):
        centers = np.asarray(centers, dtype=float)
        return np.exp(-0.5 * ((d_est - centers) / sigma)**2)



    def process_data(self):
        try:
            img_gray = np.frombuffer(self.global_camera1.data, dtype=np.uint8).reshape(
                self.global_camera1.height, self.global_camera1.width, 3
            )
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
        except Exception as e:
            self.get_logger().error(f"Error during reshaping: {e}")
            return

        max_corners = 60
        quality_level = 0.1
        min_distance = 20

        keypoints = cv2.goodFeaturesToTrack(
            img_gray,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance
        )
        image_height = self.global_camera1.height
        image_width = self.global_camera1.width

        if self.prev_gray is not None and self.prev_keypoints is not None:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, img_gray, self.prev_keypoints, None)

            good_old = self.prev_keypoints[status.flatten() == 1]
            good_new = next_pts[status.flatten() == 1]
            good_old = good_old.reshape(-1, 2)
            good_new = good_new.reshape(-1, 2)
            delta_xy = good_new - good_old  # shape (N, 2)
            # Calcul des deltas y et x
            delta_yx = np.empty((delta_xy.shape[0], 2))
            delta_yx[:, 0] = abs(delta_xy[:, 1]) # delta_y
            delta_yx[:, 1] = abs(delta_xy[:, 0]) # delta_x!!!!!!
            # Calcul des moyennes des deltas y et x
            mean_delta_y = np.mean(delta_yx[:, 0])
            mean_delta_x = np.mean(delta_yx[:, 1]) 
            var_delta_y = np.var(delta_yx[:, 0])
            var_delta_x = np.var(delta_yx[:, 1])

            # Estimation de la distance (inverse de la magnitude du flux optique moyen)!!!!!!!!
            flow_mag = np.hypot(mean_delta_x, mean_delta_y)
            flow_mag = np.clip(flow_mag, 0.0, 10.0)   
            # ou EMA
            self.flow_ema = flow_mag if getattr(self, "flow_ema", None) is None else 0.2*flow_mag + 0.8*self.flow_ema

            d_est = 1.0 / (self.flow_ema + 1e-6)  # plus de flux = plus proche

            # Codage RBF en 4 bins 
            centers = np.array([0.15, 0.30, 0.50, 0.70], dtype=float)
            sigma = 0.08  # << petit sigma = sélectif
            rbf = np.exp(-0.5 * ((d_est - centers) / sigma) ** 2)

            rbf = rbf / (rbf.sum() + 1e-9)

            bias = 0.1
            x_features = np.r_[rbf, bias]

            if self.get_parameter('log_features').get_parameter_value().bool_value:
                if (self.frame_counter % self.log_every) == 0:
                    ts = self.global_camera1_stamp.sec + self.global_camera1_stamp.nanosec * 1e-9
                    top = int(np.argmax(rbf))
                    second = float(np.partition(rbf, -2)[-2])
                    gap = float(rbf[top] - second)
                    entropy = float(-(rbf * np.log(rbf + 1e-12)).sum())
                    self.get_logger().info(
                        "[DIST_FEATS] ts=%.3f | flow=%.4f ema=%.4f | d_est=%.4f | RBF=%s "
                        "| top=%d gap=%.3f H=%.3f | min/max=%.4f/%.4f | bias=%.2f"
                        % (
                            ts, float(flow_mag), float(self.flow_ema), float(d_est),
                            np.array2string(rbf, precision=3),
                            top, gap, entropy,
                            float(rbf.min()), float(rbf.max()), bias
                        )
                    )



            # Publication 
            timestamp_sec = self.global_camera1_stamp.sec + self.global_camera1_stamp.nanosec*1e-9            
            msg = Float64MultiArray()
            msg.data = [timestamp_sec] + x_features.tolist()   # <-- ts en premier !
            self.publisher_distance_features.publish(msg)


            max_val = 10
            mean_delta_x = np.clip(mean_delta_x / max_val, 0.0, 1.0)
            mean_delta_y = np.clip(mean_delta_y / max_val, 0.0, 1.0)

            delta_yx_flat = delta_yx.flatten()
            timestamp_sec = self.global_camera1_stamp.sec + self.global_camera1_stamp.nanosec * 1e-9
            buffer_delta_yx = Float64MultiArray()
            buffer_delta_yx.data = [timestamp_sec, mean_delta_y, mean_delta_x, var_delta_y, var_delta_x]

            self.publisher_delta_yx.publish(buffer_delta_yx)
            #self.get_logger().info(f"Delta moyens publiés: Δy = {mean_delta_y:.4f}, Δx = {mean_delta_x:.4f}")


            y_old_encoded = self.encode_height(good_old, (image_height, image_width))
            y_new_encoded = self.encode_height(good_new, (image_height, image_width))
            delta_heights = (y_new_encoded - y_old_encoded)*100

            buffer_redundant = Float64MultiArray()
            buffer_redundant.data = good_new.reshape(-1).tolist()
            self.publisher_redundant.publish(buffer_redundant)

            buffer_redundant_prev = Float64MultiArray()
            buffer_redundant_prev.data = good_old.reshape(-1).tolist()
            self.publisher_redundant_prev.publish(buffer_redundant_prev)

            buffer_delta_heights = Float64MultiArray()
            timestamp_sec = self.global_camera1_stamp.sec + self.global_camera1_stamp.nanosec * 1e-9
            buffer_delta_heights.data = [timestamp_sec] + delta_heights.tolist()


            self.publisher_delta_heights.publish(buffer_delta_heights)
            #self.get_logger().info(f"Delta heights published: {delta_heights.tolist()}")
            #self.get_logger().info(f"{len(good_new)} redondant keypoints found")
            self.write_deltas_to_csv(delta_heights.tolist())



        if keypoints is not None:
            keypoints = keypoints.reshape(-1, 2)

            image_height = self.global_camera1.height
            image_width  = self.global_camera1.width

            fov_lidar, fov_camera = 20, 90     !!!!
            fov_ratio = fov_lidar / fov_camera
            center_x = image_width / 2
            half_width = (fov_ratio * image_width) / 2
            x_min, x_max = center_x - half_width, center_x + half_width

            # Seuils verticaux
            ground_threshold  = image_height * 0.75  # ignorer le bas 25%
            ceiling_threshold = image_height * 0.15  # ignorer le haut 15%

            # Filtres vectorisés (préservent la forme 2D même si vide)
            mask = (
                (keypoints[:, 0] >= x_min) &
                (keypoints[:, 0] <= x_max) &
                (keypoints[:, 1] <  ground_threshold) &
                (keypoints[:, 1] >  ceiling_threshold)
            )
            keypoints = keypoints[mask]

           # Si rien après filtres → pas d’indexation 2D, on sort proprement
            if keypoints.shape[0] == 0:
                self.get_logger().info("No keypoints after filters.")
                self.prev_keypoints = None
                # (optionnel) garder la dernière image grise
                self.prev_gray = img_gray.copy()
                return

            # Publis
            buffer_keypoints = Float64MultiArray()
            buffer_keypoints.data = keypoints.reshape(-1).tolist()
            self.publisher_keypoints.publish(buffer_keypoints)

            heights = keypoints[:, 1].astype(float)  # y en pixels
            buffer_heights = Float64MultiArray()
            buffer_heights.data = heights.tolist()
            self.publisher_heights.publish(buffer_heights)

            # Mémoires pour LK
            self.prev_gray = img_gray.copy()
            self.prev_keypoints = keypoints.reshape(-1, 1, 2).astype(np.float32)

        else:
            self.get_logger().info("No keypoints found.")
            self.prev_keypoints = None
            self.prev_gray = img_gray.copy()


    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="Visual System")
        parser.add_argument("--pos", help="Position of the frame (x:y)", default="0:0")
        parser.add_argument("--screen", help="Screen index to use", type=int, default=0)
        parser.add_argument("--debug", help="Enable debug mode", type=bool, default=False)
        return parser.parse_args()

if __name__ == "__main__":
    rclpy.init()
    node = VisualSystem()
    node.run()
    rclpy.shutdown()