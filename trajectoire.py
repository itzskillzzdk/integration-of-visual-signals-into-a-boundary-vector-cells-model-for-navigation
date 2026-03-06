import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import carla
import numpy as np
import random
import time
import csv
from std_msgs.msg import String 
import json
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
import math
import os



from cv_bridge import CvBridge

class CarlaLidarPublisher(Node):
    def __init__(self):
        super().__init__('carla_lidar_publisher')
        self.get_logger().info("Initializing CarlaLidarPublisher...")

        self.publisher_ = self.create_publisher(PointCloud2, 'carla/points', 10)
        self.sensors_publisher = self.create_publisher(String, '/vehicle/sensors', 10)
        self.heading_publisher = self.create_publisher(Float32, '/vehicle/compass_heading', 10)

        
        self.vehicle = self.init_vehicle()
        if self.vehicle is not None:
            self.lidar = self.init_lidar(self.vehicle)
            self.imu = self.init_imu(self.vehicle)  
            '''self.spawn_npc_in_front(self.vehicle, 2) '''
        else:

            self.get_logger().error("Failed to initialize vehicle.")

        self.camera_publisher = self.create_publisher(Image, '/vehicle/camera1', 10)
        self.bridge = CvBridge()

        self.raw_data_file = open('raw_lidar_data.txt', 'w')
        self.processed_data_file = open('processed_lidar_data.txt', 'w')

        self.first_frame = True


        self.get_logger().info("CarlaLidarPublisher initialized successfully.")

        self.lidar.listen(self.process_lidar)
        self.imu.listen(self.process_imu) 

        self.camera = self.init_camera(self.vehicle)
        self.camera.listen(self.process_camera)


    def init_vehicle(self):
        client = carla.Client('localhost', 2000)
        client.set_timeout(40.0)
        world = client.load_world('Town02')
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.citroen.c3')

        if not vehicle_bp: 
            self.get_logger().error("Citroen c3 blueprint not found!")
            return None

        spawn_x = 162.76
        spawn_y = 239.28
        spawn_z = 0.5
        spawn_yaw = 0.0
        spawn_point = carla.Transform(
            carla.Location(x=spawn_x, y=spawn_y, z=spawn_z),
            carla.Rotation(yaw=spawn_yaw)
        )

        try:
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            self.get_logger().info(f"Vehicle spawned manually at {spawn_point.location} with ID {vehicle.id}")
        except RuntimeError as e:
            self.get_logger().error(f"Failed to spawn vehicle: {e}")
            return None

        spectator = world.get_spectator()
        spectator.set_transform(carla.Transform(spawn_point.location + carla.Location(z=50), carla.Rotation(pitch=-90)))


        if vehicle:
            spectator = world.get_spectator()
            transform = vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))

        return vehicle

    def init_lidar(self, vehicle):
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '100')
        lidar_bp.set_attribute('rotation_frequency', '15')
        lidar_bp.set_attribute('points_per_second', '15632')
        lidar_bp.set_attribute('channels', '10')
        lidar_bp.set_attribute('horizontal_fov', '20')
        lidar_bp.set_attribute('upper_fov', '20')
        lidar_bp.set_attribute('lower_fov', '0')
        lidar_bp.set_attribute('noise_stddev', '0')
        lidar_bp.set_attribute('sensor_tick', '0.066')  

        lidar_location = carla.Location(x=0.0, z=1.8)  
        lidar_rotation = carla.Rotation(pitch=0, yaw=0, roll=0)
        lidar_transform = carla.Transform(lidar_location, lidar_rotation)

        lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
        self.get_logger().info("LiDAR sensor initialized.")
        return lidar_sensor

    def init_imu(self, vehicle):
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        imu_bp = blueprint_library.find('sensor.other.imu')
        imu_bp.set_attribute('sensor_tick', '0.1') 

        imu_location = carla.Location(0, 0, 1)
        imu_rotation = carla.Rotation(0, 0, 0)
        imu_transform = carla.Transform(imu_location, imu_rotation)

        imu_sensor = world.spawn_actor(imu_bp, imu_transform, attach_to=vehicle)
        self.get_logger().info("IMU sensor initialized.")
        return imu_sensor
    
    def init_camera(self, vehicle):
        self.get_logger().info("Attempting to initialize the camera...")

        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1920')
        camera_bp.set_attribute('image_size_y', '1080')
        camera_bp.set_attribute('fov', '90')
        camera_bp.set_attribute('sensor_tick', '0.066')

        camera_location = carla.Location(x=0.0, z=2.0) 
        camera_rotation = carla.Rotation(pitch=0)
        camera_transform = carla.Transform(camera_location, camera_rotation)

        camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        self.get_logger().info("Camera sensor initialized.")
        return camera_sensor


    def process_lidar(self, data):

        timestamp = data.timestamp  

        '''self.get_logger().info("Processing LiDAR data.")'''
        
        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))

        if self.first_frame:
            self.raw_data_file.write(str(points) + '\n')

        points = np.reshape(points, (-1, 4))

        if self.first_frame:
            self.processed_data_file.write(str(points) + '\n')
            self.first_frame = False


        pc2_msg = self.create_pointcloud2_msg(points, timestamp)
        pc2_msg.header.stamp = rclpy.time.Time(seconds=timestamp).to_msg() 
        self.publisher_.publish(pc2_msg)


    def process_imu(self, imu_data):

        compass_heading = imu_data.compass
        compass_heading_degrees = compass_heading * (180.0 / 3.141592653589793)
        heading_msg = Float32()
        heading_msg.data = compass_heading_degrees
        self.heading_publisher.publish(heading_msg)
        sensor_data = {"veh_cap_nord_deg_antitrigo": compass_heading_degrees}
        json_data = json.dumps(sensor_data)
        msg = String()
        msg.data = json_data
        self.sensors_publisher.publish(msg)

    
    def process_camera(self, image):
        try : 

            timestamp = image.timestamp  
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = np.reshape(array, (image.height, image.width, 4)) 
            rgb_image = array[:, :, :3] 
            img_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding="rgb8")
            img_msg.header.frame_id = 'camera'
            img_msg.header.stamp = rclpy.time.Time(seconds=timestamp).to_msg() 
            self.camera_publisher.publish(img_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to process camera image: {e}")


    def create_pointcloud2_msg(self, points, timestamp):
        msg = PointCloud2()
        msg.header.stamp = rclpy.time.Time(seconds=timestamp).to_msg()
        msg.header.frame_id = 'map'

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]

        msg.height = 1
        msg.width = len(points)
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * len(points)
        msg.is_dense = False
        msg.fields = fields
        msg.data = np.asarray(points, np.float32).tobytes()

        return msg
    
    def reset_vehicle(self):
        self.get_logger().info("Resetting vehicle...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()

        if self.vehicle is not None:
            self.vehicle.destroy()
            self.get_logger().info("Old vehicle destroyed.")

        self.vehicle = self.init_vehicle()
        if self.vehicle is None:
            self.get_logger().error("Failed to reinitialize vehicle.")
            return False

        self.lidar = self.init_lidar(self.vehicle)
        self.imu = self.init_imu(self.vehicle)
        self.camera = self.init_camera(self.vehicle)

        self.lidar.listen(self.process_lidar)
        self.imu.listen(self.process_imu)
        self.camera.listen(self.process_camera)

        return True
    def teleport_vehicle_to_spawn(self, spawn_transform):
        self.get_logger().info("Teleporting vehicle to spawn point...")

        self.vehicle.set_transform(spawn_transform)
        self.vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
        self.vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
        self.get_logger().info(f"Vehicle teleported to: {spawn_transform.location}")
        current_carla_time = self.get_clock().now().nanoseconds / 1e9
        '''log_dir = os.path.expanduser("~/ros_logs")
        os.makedirs(log_dir, exist_ok=True)
        teleport_log_path = os.path.join(log_dir, "teleportation_log.txt")
        with open(teleport_log_path, "a") as f:
            f.write(f"{current_carla_time}\n")'''




    
    def drive_to_target(self, target_location, stop_distance=25, speed=0.4):
       self.get_logger().info(f"Driving towards target at {target_location}...")

       while rclpy.ok():
            current_transform = self.vehicle.get_transform()
            current_location = current_transform.location

            dx = target_location.x - current_location.x
            dy = target_location.y - current_location.y
            distance = (dx**2 + dy**2)**0.5
            #self.get_logger().info(f"Current vehicle location: x={current_location.x:.2f}, y={current_location.y:.2f}, distance to target={distance:.2f} m")

            if distance < stop_distance:
                self.get_logger().info(f"Reached target (distance: {distance:.2f} m). Stopping.")
                self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                break


            angle_to_target = np.arctan2(dy, dx)
            yaw_rad = np.radians(current_transform.rotation.yaw)
            angle_diff = (angle_to_target - yaw_rad + np.pi) % (2 * np.pi) - np.pi
            steer = np.clip(angle_diff, -1.0, 1.0)

            control = carla.VehicleControl(throttle=speed, steer=steer, brake=0.0)
            self.vehicle.apply_control(control)
   
            time.sleep(0.05)

        

    def run(self, repetitions=100):
        spawn_points = [
            #carla.Transform(carla.Location(x=150.76, y=239.28, z=0.5), carla.Rotation(yaw=0.0)),  # 50m
            carla.Transform(carla.Location(x=180.0, y=239.28, z=0.5), carla.Rotation(yaw=0.0)),   # 30m
            carla.Transform(carla.Location(x=195.0, y=239.28, z=0.5), carla.Rotation(yaw=0.0)),   # 15m
            #carla.Transform(carla.Location(x=27.15, y=105.55, z=0.5), carla.Rotation(yaw=-180.0)),
        ]
        target = carla.Location(x=208.64, y=239.77, z=0.26)       
        for i in range(repetitions):
            self.get_logger().info(f"--- Starting repetition {i+1}/{repetitions} ---")
            '''if not self.reset_vehicle():
                self.get_logger().error(f"Repetition {i+1} failed due to vehicle reset error.")
                break'''
         
            spawn = random.choice(spawn_points)
            self.teleport_vehicle_to_spawn(spawn)
         
            time.sleep(1.0)  
            if abs(spawn.location.x - 27.15) < 0.1:
                self.drive_to_target(target, stop_distance=40)
            else:
                self.drive_to_target(target, stop_distance=17)
            self.get_logger().info(f"--- Repetition {i+1}/{repetitions} completed ---")
            time.sleep(1.0)
        self.get_logger().info("All repetitions completed. Node will keep publishing sensor data...")

        
        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.1)
        except KeyboardInterrupt:
            self.get_logger().info('Node interrupted by user.')


    def destroy_node(self):
        self.raw_data_file.close()
        self.processed_data_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    carla_lidar_publisher = CarlaLidarPublisher()
    if carla_lidar_publisher.vehicle is not None:
        try:
            carla_lidar_publisher.run(repetitions=100)
        finally:
            carla_lidar_publisher.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()

