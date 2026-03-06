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
    """ Normalize angle to be within the range [0, 2π]. """
    return angle % (2 * np.pi)

def callback_SpdC_activity(data):
    global global_SpdC_activity, global_SpdC_activity_flag

    for i in range(0, len(data.data), 17):
        angle = normalize_angle(data.data[i])
        ratesN = data.data[i+1:i+5]
        ratesS = data.data[i+5:i+9]
        ratesE = data.data[i+9:i+13]
        ratesW = data.data[i+13:i+17]
       

        for j, rate in enumerate(ratesN):
                rate = np.clip(rate, 0, 1)  # dans le callback

                global_SpdC_activity["North"][j].append(rate)
                if len(global_SpdC_activity["North"][j]) > window_size:
                    global_SpdC_activity["North"][j].pop(0)

           # print(f"Angle {angle} categorized as {"North"} with rates {ratesN}.")
        
        

        for j, rate in enumerate(ratesS):
                rate = np.clip(rate, 0, 1) 
                global_SpdC_activity["South"][j].append(rate)
                if len(global_SpdC_activity["South"][j]) > window_size:
                    global_SpdC_activity["South"][j].pop(0)

                #print(f"Angle {angle} categorized as {"South"} with rates {ratesS}.")
       
        
        for j, rate in enumerate(ratesE):
                rate = np.clip(rate, 0, 1)  # dans le callback

                global_SpdC_activity["East"][j].append(rate)
                if len(global_SpdC_activity["East"][j]) > window_size:
                    global_SpdC_activity["East"][j].pop(0)
            #print(f"Angle {angle} categorized as {"East"} with rates {ratesE}.")
        
        
        for j, rate in enumerate(ratesW):
                rate = np.clip(rate, 0, 1)  # dans le callback

                global_SpdC_activity["West"][j].append(rate)
                if len(global_SpdC_activity["West"][j]) > window_size:
                    global_SpdC_activity["West"][j].pop(0)
          #  print(f"Angle {angle} categorized as {"West"} with rates {ratesE}.")
        

    
    global_SpdC_activity_flag = True

node.create_subscription(Float64MultiArray, 'reference_bvc', callback_SpdC_activity, 10)

def ros_spin():
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=1)

ros_thread = threading.Thread(target=ros_spin)
ros_thread.start()

pygame.init()
pygame_screenSize = (800, 800)
pygame_screen = pygame.display.set_mode(pygame_screenSize)
pygame.display.set_caption("BVC Activations - reference(LIDAR)")


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

            ax.set_ylim(1e-10, 0.01)
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