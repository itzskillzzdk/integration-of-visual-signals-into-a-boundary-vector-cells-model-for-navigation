import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import carla
import math 
from collections import deque

class LidarDistanceCalculator(Node):
    def __init__(self):
        super().__init__('lidar_distance_calculator')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/carla/points',
            self.listener_callback,
            10)

        self.publisher_ = self.create_publisher(Float32MultiArray, '/distang_topic', 10)  
        self.activity_publisher = self.create_publisher(Float32MultiArray, '/activity_matrix', 10)
        self.compass_publisher = self.create_publisher(Float32MultiArray, 'vehicle_compass', 10)      
        self.OpenAngle = np.radians(20)
        self.num_channels = 10
        self.delta_y= self.OpenAngle / self.num_channels
        self.distance_buffers = [deque(maxlen=10) for _ in range(self.num_channels)]  # Example window size of 10
        self.vehicle = self.connect_to_vehicle()
        self.compass_value = None 
        self.imu_sensor = self.get_imu_sensor()
        
        if self.imu_sensor : 
            self.imu_sensor.listen(self.process_imu)

    def connect_to_vehicle(self):
        client = carla.Client('localhost', 2000)
        client.set_timeout(40.0)
        world = client.get_world()
        
        vehicle = None
        for actor in world.get_actors():
            if actor.type_id.startswith('vehicle.'):
                vehicle = actor
                self.get_logger().info(f"Connected to vehicle: {vehicle.id}.")
                break

        if not vehicle:
            self.get_logger().error("No vehicle found.")
        
        return vehicle
    
    def get_imu_sensor(self):
        imu_sensor = None
        for actor in self.vehicle.get_world().get_actors():
            if actor.type_id.startswith('sensor.other.imu'):
                imu_sensor = actor
                break

        if imu_sensor is None:
            self.get_logger().error("IMU sensor not found.")
        return imu_sensor
    
    def process_imu(self, imu_data):
        self.compass_value = imu_data.compass
        msg = Float32MultiArray()
        msg.data = [self.compass_value] 
        self.compass_publisher.publish(msg)
        self.get_logger().info(f"Updated Compass heading: {np.degrees(self.compass_value):.2f} .")

    def get_quadrant(self, angle_deg):
        if -45 <= angle_deg < 45:
            return 0  # nord
        elif 45 <= angle_deg < 135:
            return 1  # east
        elif 135 <= angle_deg < 225:
            return 2  # sud
        else:
            return 3  # west
    def get_distance_bin(self, distance):
        if distance < 7:
            return 0
        elif distance < 10:
            return 1
        elif distance < 20:
            return 2
        elif distance < 30:
            return 3
        else:
            return None 
        

    def get_visible_quadrants(self, heading_rad, fov_deg=90):
    # Azimuts allocentriques des 4 quadrants (en radians)
        quadrant_dirs = {
            0: 0,                # North
            1: np.pi / 2,        # East
            2: np.pi,            # South
            3: 3 * np.pi / 2     # West
        }   

        visible_quadrants = []
        fov_half = np.radians(fov_deg / 2)
        for q, az in quadrant_dirs.items():
            angle_diff = np.arctan2(np.sin(az - heading_rad), np.cos(az - heading_rad))
            if abs(angle_diff) <= fov_half:
                visible_quadrants.append(q)

        return visible_quadrants

    def get_distance_weights(self, distance):
        bins = [7.0, 10.0, 20.0, 30.0]  
        sigma = 0.3
        weights = np.exp(-((distance - np.array(bins)) ** 2) / (2 * sigma ** 2))
        total = np.sum(weights)

        if total < 1e-8:
        # Cas rare : poids négligeables, évite NaN
            return np.zeros_like(weights)
        return weights / total
        
    
    def get_quadrant_weights(self, angle_deg):
        quadrants = [0, 90, 180, 270 ]  # centres des 4 quadrants
        sigma = 30.0  # en degrés
        diffs = np.array([(angle_deg - q + 360) % 360 for q in quadrants])
        diffs = np.minimum(diffs, 360 - diffs)  
        weights = np.exp(- (diffs ** 2) / (2 * sigma ** 2))
        return weights / np.sum(weights)



        
    
    def listener_callback(self, msg):
        #self.get_logger().info(" Received a LiDAR frame!")
        if self.compass_value is None:
           #self.get_logger().info("Compass value not available yet.")
           return
        carla_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        points = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        
        points_array = np.array(list(points), dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32)])
        points_array = np.stack([points_array['x'], points_array['y'], points_array['z']], axis=-1)
        if points_array.size == 0:
            #self.get_logger().info("No points received in this frame.")
            return
        
        if points_array.ndim != 2 or points_array.shape[1] != 3:
            #self.get_logger().error(f"Unexpected points array shape: {points_array.shape}")
            return
        
    #    self.process_points(points_array , self.compass_value)

    #def process_points(self, points_array , compass_value):
       
        min_distances = {}
        min_distances_counter = {}
        activity_matrix = np.zeros((4, 4), dtype=float)
        for point in points_array :
            x, y, z = point
            allocentric_azimuth = (np.arctan2(y, x) + self.compass_value) % (2 * np.pi)
            relative_azimuth = (allocentric_azimuth - self.compass_value + np.pi) % (2 * np.pi) - np.pi  # [-π, π]

            if not (-self.OpenAngle / 2 <= relative_azimuth <= self.OpenAngle / 2):
                continue

            direction_index = int((relative_azimuth + self.OpenAngle / 2) / self.delta_y)
            distance = math.sqrt(x**2 + y**2 + z**2)

            if direction_index not in min_distances or distance < min_distances[direction_index]:
                min_distances[direction_index] = distance


        for direction_index, dist_val in min_distances.items():
            fov_deg = 30  # ton vrai FOV
            angle_step = fov_deg / self.num_channels
            start_angle = np.degrees(self.compass_value) - fov_deg / 2
            direction_angle = (start_angle + direction_index * angle_step) % 360    

            angle_deg = direction_angle
            quad_index = self.get_quadrant(angle_deg)
            bin_index = self.get_distance_bin(dist_val)
            weights = self.get_distance_weights(dist_val)
            threshold = 1e-2
            for bin_idx, weight in enumerate(weights):
                if weight < threshold:
                    continue
                activity_matrix[quad_index][bin_idx] += weight
                #self.get_logger().debug(f"[ACTIVITY] Quad={quad_index}, Bin={bin_idx}, Distance={distance:.2f}, Weight={weight:.4f}")


            '''if bin_index is not None:
                bins = [1.0, 4.0, 7.0, 10.0]  # centres des bins
                sigma = 1.0
                center_distance = bins[bin_index]
                activity = math.exp(-((distance - center_distance) ** 2) / (2 * sigma ** 2))

                self.get_logger().debug(f"[ACTIVITY] Direction={direction_index}, Angle={angle_deg:.2f}°, Distance={distance:.2f}, Activity={activity:.4f}")

                activity_matrix[quad_index][bin_index] += activity'''

        epsilon = 1e-6
        activity_matrix += np.random.uniform(0, epsilon, size=activity_matrix.shape)


        max_val = np.max(activity_matrix)
        if max_val > 1e-6:  
            activity_matrix = activity_matrix / max_val
        else:
            activity_matrix = np.full_like(activity_matrix, epsilon)


        # Affiche la matrice normalisée
        # self.get_logger().info(f"Matrice d'activité normalisée:\n{activity_matrix}")

        activity_msg = Float32MultiArray()
        activity_msg.data.append(float(carla_timestamp)) 
        activity_msg.data.extend(activity_matrix.flatten().tolist()) 
        self.activity_publisher.publish(activity_msg)
        #self.get_logger().info("Published activity matrix.")
        
 

        msg = Float32MultiArray()
        msg.data.append(carla_timestamp)  
        fov_deg = 30  # même valeur que plus haut
        angle_step = fov_deg / self.num_channels
        start_angle = np.degrees(self.compass_value) - fov_deg / 2

        for direction_index in range(self.num_channels):
            direction_angle = (start_angle + direction_index * angle_step) % 360
            # self.get_logger().info(f"Direction {direction_index:2d} | Angle: {direction_angle:6.2f}° | Distance: {distance:6.2f} m")


            if direction_index in min_distances:
                min_distance = min_distances[direction_index]
                msg.data.extend([direction_angle, min_distance])
                self.get_logger().info(f"Published: Angle: {direction_angle:.2f} degrees, Distance: {min_distance:.2f} meters" )
            else:
                self.get_logger().info( f"Direction Angle: {direction_angle:.2f} degrees, Min Distance: inf " )

        self.publisher_.publish(msg)



def main(args=None):
    rclpy.init(args=args)
    lidar_distance_calculator = LidarDistanceCalculator()
    rclpy.spin(lidar_distance_calculator)

    lidar_distance_calculator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
