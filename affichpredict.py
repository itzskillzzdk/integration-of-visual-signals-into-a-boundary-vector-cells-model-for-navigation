import sys
import threading
import pygame
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


rclpy.init()
node = Node('BVC_displayer')


directions = ["North", "South", "East", "West"]
cell_names = ["d=2", "d=6", "d=8", "d=12"]
colors = ["blue", "orange", "green", "red"]
window_size = 8  
global_SpdC_activity = {direction: [[] for _ in range(4)] for direction in directions}
global_SpdC_activity_flag = False


angle_ranges = {
    "North": (0, 1.57),  # 0 to π/2
    "East": (1.57, 3.14),  # π/2 to π
    "South": (3.14, 4.71),  # π to 3π/2
    "West": (4.71, 6.28)  
}

def normalize_angle(angle):
    return angle % (2 * np.pi)

def callback_SpdC_activity(data):
    global global_SpdC_activity, global_SpdC_activity_flag
    matrix = np.array(data.data).reshape((4, 4))  # 4 quadrants × 4 distances


    for i, direction in enumerate(directions):  # i = quadrant index
        for j in range(4):  # j = distance bin index
            rate = float(np.clip(matrix[i][j], 0, 1))
            global_SpdC_activity[direction][j].append(rate)
            if len(global_SpdC_activity[direction][j]) > window_size:
                global_SpdC_activity[direction][j].pop(0)

    global_SpdC_activity_flag = True
        

    
node.create_subscription(Float64MultiArray, '/predicted_bvc', callback_SpdC_activity, 10)

def ros_spin():
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=1)

ros_thread = threading.Thread(target=ros_spin)
ros_thread.start()

pygame.init()
pygame_screenSize = (800, 800)
pygame_screen = pygame.display.set_mode(pygame_screenSize)
pygame.display.set_caption("BVC Activations - Predicted(Vision)")


fig, axs = plt.subplots(4, 1, figsize=(8, 8), dpi=100)
direction_labels = ["North", "South","East",  "West"]

# Main loop
running = True
while running:
    rclpy.spin_once(node, timeout_sec=0)  # Check for incoming messages

    if global_SpdC_activity_flag:
        for ax, direction in zip(axs, directions):
            ax.clear()

            angle_range = angle_ranges[direction] 
            angle_min = np.degrees(angle_range[0])
            angle_max = np.degrees(angle_range[1])
            ax.set_title(f"Direction: {direction} ({angle_min:.1f}° to {angle_max:.1f}°)")

            ax.set_ylim(0, 1)
            ax.set_ylabel("Activation")
            ax.set_xlabel("Cell Distance")
    
            avg_activities = [
                np.mean(global_SpdC_activity[direction][j]) if global_SpdC_activity[direction][j] else 0
                for j in range(4)
            ]

            ax.bar(cell_names, avg_activities, color=colors)

        global_SpdC_activity_flag = False  # Reset flag after plotting

        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.buffer_rgba().tobytes()
        surf = pygame.image.fromstring(raw_data, canvas.get_width_height(), "RGBA")
        pygame_screen.blit(surf, (0, 0))
        pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
rclpy.shutdown()

if ros_thread.is_alive():
    ros_thread.join()
