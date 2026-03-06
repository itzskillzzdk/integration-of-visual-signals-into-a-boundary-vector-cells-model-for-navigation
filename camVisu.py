import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
# from cv_bridge import CvBridge
import numpy as np
import pygame
import cv2
from message_filters import Subscriber, ApproximateTimeSynchronizer


class CameraKeypointViewer(Node):
    def __init__(self):
        super().__init__('camera_keypoint_viewer')

        # ROS 2 - Souscriptions
        self.image_subscriber = self.create_subscription(
            Image, '/vehicle/camera1', self.image_callback, 10)

        self.keypoints_subscriber = self.create_subscription(
            Float64MultiArray, '/visualSystem/keypoints', self.keypoints_callback, 10)
        

        self.redundant_keypoints_subscriber = self.create_subscription(
            Float64MultiArray, '/visualSystem/redundant_keypoints', self.redundant_keypoints_callback, 10)

        self.image_sub = Subscriber(self, Image, '/vehicle/camera1')
        self.redundant_kp_sub = Subscriber(self, Float64MultiArray, '/visualSystem/redundant_keypoints')
        self.redundant_previous_keypoints_subscriber = self.create_subscription(
            Float64MultiArray, '/visualSystem/redundant_keypoints_previous', self.redundant_previous_keypoints_callback, 10)

        self.redundant_previous_keypoints = []
        #self.kp_sub = Subscriber(self, Float64MultiArray, '/visualSystem/keypoints')

        #self.ts = ApproximateTimeSynchronizer([self.image_sub, self.kp_sub], queue_size=10, slop=0.1, allow_headerless=True)
        self.ts = ApproximateTimeSynchronizer([self.image_sub, self.redundant_kp_sub], queue_size=10, slop=0.1, allow_headerless=True)

        self.ts.registerCallback(self.synced_callback)




        # Convertisseur ROS <-> OpenCV
        # self.bridge = CvBridge()

        # Stockage des points clés
        self.keypoints = []
        self.redundant_keypoints = []


        # Initialisation de pygame
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("ROS 2 Keypoint Viewer")

        # Stockage de l’image
        self.image = None

        #self.get_logger().info("Camera Keypoint Viewer Node Started")

    def synced_callback(self, image_msg, redundant_keypoints_msg):
        self.image_callback(image_msg)
        self.redundant_keypoints_callback(redundant_keypoints_msg)


    def keypoints_callback(self, msg):
 
        if not msg.data:
            #self.get_logger().warn("Received empty keypoints message!")
            return

        self.keypoints = [(msg.data[i], msg.data[i+1]) for i in range(0, len(msg.data), 2)]
        #self.get_logger().info(f"Received {len(self.keypoints)} keypoints: {self.keypoints[:5]}")

    def redundant_keypoints_callback(self, msg):
        """
        Callback pour recevoir la liste des points clés redondants (x, y).
        """
        if not msg.data:
            #self.get_logger().warn("Received empty redundant keypoints message!")
            return

        self.redundant_keypoints = [(msg.data[i], msg.data[i+1]) for i in range(0, len(msg.data), 2)]

        #self.get_logger().info(f"Received {len(self.redundant_keypoints)} redundant keypoints: {self.redundant_keypoints[:5]}")

    def redundant_previous_keypoints_callback(self, msg):
        if not msg.data:
            return
        self.redundant_previous_keypoints = [(msg.data[i], msg.data[i+1]) for i in range(0, len(msg.data), 2)]

    # def image_callback(self, msg):
    #     """
    #     Callback pour recevoir l'image et la stocker.
    #     """
    #     self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')


    #     '''self.image = np.rot90(self.image)  
    #     self.image = np.flipud(self.image)  '''

    #     # Redimensionner l’image à la taille de l’écran
    #     self.image = cv2.resize(self.image, (self.screen_width, self.screen_height))

    def image_callback(self, msg):
        """
        Callback manuel sans cv_bridge
        """
        img_array = np.frombuffer(msg.data, dtype=np.uint8)
        self.image = img_array.reshape((msg.height, msg.width, 3))

        if msg.encoding == 'rgb8':
            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        self.image = cv2.resize(self.image, (self.screen_width, self.screen_height))

    def draw_keypoints_pygame(self, surface, keypoints, color=(255, 50, 50), size=10):
        """
        Dessine une croix rouge sur les points clés dans une image affichée avec pygame.
        """
        original_width = 1920
        original_height = 1080
        for x, y in keypoints:
                # Rééchelonner les coordonnées des points clés à la taille de l'affichage
            x = int((x / original_width) * self.screen_width)
            y = int((y / original_height) * self.screen_height)
            
            if 0 <= x < self.screen_width and 0 <= y < self.screen_height:
                pygame.draw.line(surface, color, (x - size, y), (x + size, y), 2)  # Ligne horizontale
                pygame.draw.line(surface, color, (x, y - size), (x, y + size), 2)  # Ligne verticale

    def render(self):
        """
        Affiche l'image en temps réel avec pygame et dessine les points clés.
        """
        if self.image is not None:
            image_copy = self.image.copy()

            # Convertir pour pygame (OpenCV: BGR -> RGB et format pygame)
            image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
            surface = pygame.surfarray.make_surface(np.rot90(image_rgb))  # Nouvelle rotation inutile '
            surface = pygame.transform.flip(surface, True, False)

            # Afficher avec pygame
            self.screen.blit(surface, (0, 0))

            # Dessiner les points clés sous forme de croix rouge
            self.draw_keypoints_pygame(self.screen, self.redundant_keypoints, color=(255, 50, 50), size=10)
            self.draw_keypoints_pygame(self.screen, self.redundant_previous_keypoints, color=(50, 50, 255), size=10)
            self.draw_keypoints_pygame(self.screen, self.keypoints, color=(0, 255, 0), size=8)


            font = pygame.font.SysFont(None, 30)
            text_surface = font.render(f"Redundant Keypoints: {len(self.redundant_keypoints)}", True, (255, 255, 255))
            self.screen.blit(text_surface, (10, 10))


            pygame.display.flip()

    def run(self):
        """
        Boucle principale avec gestion des événements Pygame.
        """
        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.05)
                self.render()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
        except KeyboardInterrupt:
            self.get_logger().info("Shutting down Camera Keypoint Viewer")
        finally:
            self.cleanup()

    def cleanup(self):
        """Stoppe proprement Pygame."""
        pygame.quit()

def main(args=None):
    rclpy.init(args=args)
    node = CameraKeypointViewer()
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



