import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64MultiArray
import matplotlib.pyplot as plt
import numpy as np
import threading

class ResultPlotter(Node):
    def __init__(self):
        super().__init__('result_plotter')

        self.sub_target = self.create_subscription(Float32MultiArray, '/activity_matrix', self.target_callback, 10)
        self.sub_pred = self.create_subscription(Float64MultiArray, '/predicted_bvc', self.pred_callback, 10)

        self.target_matrix = np.zeros((4,4))
        self.pred_matrix = np.zeros((4,4))
        self.directions = ["Nord", "Est", "Sud", "Ouest"]
        self.distances = ["Proche", "Moyen", "Loin", "Très Loin"]

        print("Noeud de plot prêt. Appuyer sur [ENTREE] dans ce terminal pour sauvegarder la figure.")

    def target_callback(self, msg):
        data = np.array(msg.data[1:])
        if data.size == 16:
            self.target_matrix = data.reshape((4,4))

    def pred_callback(self, msg):
        data = np.array(msg.data)
        if data.size == 16:
            self.pred_matrix = data.reshape((4,4))
    
    def save_figure(self):
        if np.all(self.target_matrix == 0) and np.all(self.pred_matrix == 0):
            print("Attention: les matrices sont vides ou nulles.")
        fig, axes = plt.subplots(1,2,figsize=(12,5))
        cmap = 'viridis'
        vmin, vmax = 0, 1
        im1 = axes[0].imshow(self.target_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0].set_title("Vérité Terrain (LIDAR)")
        axes[0].set_xticks(np.arange(4))
        axes[0].set_yticks(np.arange(4))
        axes[0].set_xticklabels(self.distances)
        axes[0].set_yticklabels(self.directions)

        im2 = axes[1].imshow(self.pred_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1].set_title("Prédiction (Vision seule)")
        axes[1].set_xticks(np.arange(4))
        axes[1].set_yticks(np.arange(4))
        axes[1].set_xticklabels(self.distances)
        axes[1].set_yticklabels([])

        fig.colorbar(im1, ax=axes.ravel().tolist(), label="Niveau d'activation")

        filename = "resultat_bvc_comparaison.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Figure sauvegardée sous : {filename}")
        plt.close(fig)

def input_thread(node):
    while rclpy.ok():
        input()
        node.save_figure()

def main(args=None):
    rclpy.init(args=args)
    plotter = ResultPlotter()

    t = threading.Thread(target=input_thread, args=(plotter,), daemon=True)
    t.start()

    rclpy.spin(plotter)
    plotter.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
