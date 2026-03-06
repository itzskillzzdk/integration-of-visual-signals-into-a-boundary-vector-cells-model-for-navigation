import numpy as np
import rclpy
from rclpy.node import Node
import pickle
from std_msgs.msg import Float64MultiArray, Float32MultiArray ,Float32
import sys
sys.path.append("/home/mariem/Bureau/ros2_ws/src/bvc/bvc/bvcCamLearn/src")  
from src.computation.NL_LMS import LMS  
import os
import datetime

log_dir = os.path.expanduser("~/ros_logs")  
os.makedirs(log_dir, exist_ok=True)  
log_file = os.path.join(log_dir, "lidar_camera_errors.log") 
weights_file = os.path.join(log_dir, "lms_model.pkl")


def save_error_to_log(timestamp, error_bvc):
    with open(log_file, "a") as f:
        f.write(f"{timestamp},{error_bvc}\n")


'''log_dir = os.path.expanduser("~/ros_logs")
weights_file = os.path.join(log_dir, "lms_model.pkl")

with open(weights_file, 'rb') as f:
    lms = pickle.load(f)

print("Poids LMS :", lms.w_ij)'''





class BVC:
    def __init__(self, PDist, PTheta, sigma_dist, sigma_theta):
        self.PDist = PDist 
        self.PTheta = PTheta
        self.sigma_dist = sigma_dist
        self.sigma_theta = sigma_theta

        
class LidarCameraLMS(Node):
    def __init__(self):
        super().__init__('lidar_camera_lms')
        with open(log_file, "w") as f:
            f.write("timestamp,error_bvc\n") 
        
        self.lms = LMS(n_neurons=16, input_size=2) 
       
        if os.path.isfile(weights_file):
            try:
                with open(weights_file, 'rb') as f:
                    self.lms = pickle.load(f)
                #self.get_logger().info(f"LMS chargé depuis {weights_file}")
            except Exception as e:
                self.get_logger().warn(f"Erreur chargement LMS : {e}")
        self.subscription_camera = self.create_subscription(
            Float64MultiArray, '/visualSystem/redundant_deltas', self.camera_callback, 10)
        self.subscription_lidar = self.create_subscription(
            Float32MultiArray, '/distang_topic', self.lidar_callback, 10)
        
        self.subscription_matrix = self.create_subscription(
            Float32MultiArray,
            '/activity_matrix',
            self.activity_matrix_callback,
            10
        )

        self.subscription_heading = self.create_subscription(
            Float32,
            '/vehicle/compass_heading',
            self.heading_callback,
            10
        )


        self.publisher_predicted_vision = self.create_publisher(Float64MultiArray, 'predicted_vision', 10)
        self.publisher_predicted_bvc = self.create_publisher(Float64MultiArray, 'predicted_bvc', 10)
        #stockage des moyennes pour les deltas
        self.moyennes_delta_az_list = []
        self.moyennes_delta_h_list = []
        self.delta_h_vs_distance = [] 
        self.camera_timestamps = []

        self.lidar_data = []  
        self.camera_data = []  
        self.all_predictions = []
        self.vehicle_heading = None
        self.last_lidar_ts = None
        self.bvc_matrix_buffer = [] 
        self.BVCs = {
            "North": [BVC(d, 0, sigma, np.pi / 8) for d, sigma in [(2.0, 0.5), (6.0, 1.0), (8.0, 1.2), (12.0, 1.5)]],
            "South": [BVC(d, np.pi, sigma, np.pi / 8) for d, sigma in [(2.0, 0.5), (6.0, 1.0), (8.0, 1.2), (12.0, 1.5)]],
            "East":  [BVC(d, np.pi / 2, sigma, np.pi / 8) for d, sigma in [(2.0, 0.5), (6.0, 1.0), (8.0, 1.2), (12.0, 1.5)]],
            "West":  [BVC(d, 3 * np.pi / 2, sigma, np.pi / 8) for d, sigma in [(2.0, 0.5), (6.0, 1.0), (8.0, 1.2), (12.0, 1.5)]]
        }

    def get_closest_direction(self, azimuth):
        """ Trouve la direction cardinale (Nord, Sud, Est, Ouest) la plus proche en fonction de l'azimut. """
        directions = {
            "North": 0,
            "East": np.pi / 2,
            "South": np.pi,
            "West": 3 * np.pi / 2
        }

        # Trouver la direction avec la plus petite différence d'angle
        closest_direction = min(directions, key=lambda dir: 
            abs(np.arctan2(np.sin(directions[dir] - azimuth), np.cos(directions[dir] - azimuth))))
        return closest_direction
    
    def camera_callback(self, msg):
        deltas = msg.data
        timestamp = deltas[0]  
        deltas = deltas[1:]    

        visual_frame = []


        for i in range(0, len(deltas), 3):
            az = np.radians(deltas[i]) % (2 * np.pi)
            delta_az = deltas[i + 1]
            delta_h = deltas[i + 2]
            visual_frame.append((az, delta_az, delta_h))
   
        self.camera_data.append((timestamp, visual_frame))
        self.camera_data = self.camera_data[-50:]


        if self.lidar_data:
            lidar_timestamps = np.array([ts for ts, _ in self.lidar_data])
            time_diffs = np.abs(lidar_timestamps - timestamp)
            closest_idx = np.argmin(time_diffs)

            closest_lidar_frame = self.lidar_data[closest_idx][1]  
            distances = np.array([dist for _, dist in closest_lidar_frame])

            min_distance = np.min(distances)

            for _, _, delta_h in visual_frame:
                self.delta_h_vs_distance.append((min_distance, delta_h))

        with open(os.path.join(log_dir, "camera_deltas.csv"), "a") as f:
            for _, _, delta_h in visual_frame:
                f.write(f"{timestamp},{min_distance},{delta_h}\n")

    

        #self.get_logger().info(f" Nouvelle frame caméra  {timestamp}")
        if self.lidar_data:
            self.train_lms(timestamp, visual_frame)

    def heading_callback(self, msg):
        # Le heading est en degrés, on le convertit en radians pour rester cohérent
        self.vehicle_heading = np.radians(msg.data) % (2 * np.pi)
        #self.get_logger().info(f"Vehicle heading mis à jour : {msg.data:.2f}° → {self.vehicle_heading:.2f} rad")

    def activity_matrix_callback(self, msg):
        try:
            data = msg.data
            timestamp = data[0]  # timestamp Carla
            matrix = np.array(data[1:]).reshape((4, 4))  # 16 valeurs après le timestamp

            self.bvc_matrix_buffer.append((timestamp, matrix))
            self.bvc_matrix_buffer = self.bvc_matrix_buffer[-100:]
            self.last_bvc_ts = self.get_clock().now().nanoseconds / 1e9  
            #self.get_logger().info(f" Matrice BVC reçue @ {timestamp:.2f}s")
        except:
            self.get_logger().warn("Format de matrice invalide reçu.")


    def lidar_callback(self, msg):
        ''' Callback pour traiter les données du LiDAR et ne conserver que les points dans le FoV de la caméra '''
        self.last_lidar_ts = self.get_clock().now().nanoseconds
        data = msg.data
        if len(data) < 3:
            return  

        timestamp = data[0]
        lidar_entries = list(zip(data[1::2], data[2::2]))

        # Conversion des azimuts LiDAR en radians et normalisation
        lidar_angles = np.radians([az for az, _ in lidar_entries]) % (2 * np.pi)
        lidar_entries_radians = [(np.radians(az) % (2 * np.pi), dist) for az, dist in lidar_entries]


        #self.get_logger().info(f" Converted LiDAR Azimuths (Radians): {lidar_angles}")
        #self.get_logger().info(f" Converted LiDAR Azimuths (Degrees): {np.degrees(lidar_angles)}")

        self.lidar_data.append((timestamp, lidar_entries_radians))
        self.lidar_data = self.lidar_data[-100:]

        #self.get_logger().info(f"Nouvelle frame LiDAR @ {timestamp}")
        # Log des azimuts LiDAR retenus après filtrage
        #self.get_logger().info(f" LiDAR Azimuths After Filtering (FoV 90°): {np.degrees([az for _, az, _ in filtered_lidar_data])}°")
    def save_lms(self):
        import pickle
        try:
            with open(weights_file, 'wb') as f:
                pickle.dump(self.lms, f)
            self.get_logger().info(f"LMS sauvegardé dans {weights_file}")
        except Exception as e:
            self.get_logger().error(f"Erreur sauvegarde LMS : {e}")

    def preprocess_input(self, visual_frame):
        if not visual_frame:
            return np.array([])
        moyenne_delta_az = np.mean([delta_az for _, delta_az, _ in visual_frame])
        moyenne_height = np.mean([delta_h for _, _, delta_h in visual_frame])
        self.moyennes_delta_az_list.append(moyenne_delta_az)
        self.moyennes_delta_h_list.append(moyenne_height)
        '''timestamp = visual_frame[0][0]   

        self.camera_timestamps.append(timestamp)

        if self.lidar_data:
            lidar_timestamps = np.array([ts for ts, _ in self.lidar_data])
            time_diffs = np.abs(lidar_timestamps - timestamp)
            closest_idx = np.argmin(time_diffs)

            closest_lidar_frame = self.lidar_data[closest_idx][1]  # liste de (azimuth, distance)
            distances = np.array([dist for _, dist in closest_lidar_frame])

            min_distance = np.min(distances)
        else:
            pass

        with open(os.path.join(log_dir, "camera_deltas.csv"), "a") as f:
            f.write(f"{timestamp},{min_distance},{moyenne_delta_az},{moyenne_height}\n")'''
   

        x = np.array([moyenne_delta_az, moyenne_height])

        x_min = np.array([-1.0, -1.0])
        x_max = np.array([1.0, 1.0])

        x = (x - x_min) / (x_max - x_min + 1e-8)
        return np.clip(x, 0.0, 1.0)



    def train_lms(self, cam_ts, visual_frame ):

        # Trouver la donnée LiDAR avec le timestamp le plus proche


        Δt_max = 0.2  # par exemple 200 ms
        timestamps = np.array([ts for ts, _ in self.bvc_matrix_buffer])
        time_diffs = np.abs(timestamps - cam_ts)
        now = self.get_clock().now().nanoseconds / 1e9  # Temps actuel en secondes
   
        if not self.bvc_matrix_buffer:
            self.get_logger().warn(f"Pas de BVC dans le buffer → prédiction seule activée.")

        timestamps = np.array([ts for ts, _ in self.bvc_matrix_buffer])
        time_diffs = np.abs(timestamps - cam_ts)
        closest_idx = np.argmin(time_diffs)
        if time_diffs[closest_idx] > Δt_max:
            self.get_logger().warn(f"Aucune BVC assez proche de t_cam={cam_ts:.3f}s (Δt = {time_diffs[closest_idx]:.3f}s) → prédiction seule activée.")
            self.predict_only(visual_frame)
            return
            

        
        x = np.array([[delta_az, delta_h] for _, delta_az, delta_h in visual_frame]).flatten()
        

        # Récupère la matrice BVC la plus proche en temps du timestamp caméra
        timestamps = np.array([ts for ts, _ in self.bvc_matrix_buffer])
        time_diffs = np.abs(timestamps - cam_ts)
        closest_idx = np.argmin(time_diffs)
        closest_matrix = self.bvc_matrix_buffer[closest_idx][1]

        #self.get_logger().info(f" Matched BVC matrix with Δt = {time_diffs[closest_idx]:.3f}s")
        self.get_logger().info(f"Camera TS: {cam_ts:.3f}, Closest BVC TS: {timestamps[closest_idx]:.3f}, Δt = {time_diffs[closest_idx]:.3f}")

        y = closest_matrix.flatten()

        if x.size == 0 or y.size == 0:
            self.get_logger().warn(" Aucun point associé Vision ↔ LiDAR, x ou y vide avant padding")
        else:
            self.get_logger().info(f" Nombre de correspondances Vision/LiDAR: {len(x)}")


        x = self.preprocess_input(visual_frame)
        print(f"[LMS] Input x: {x}")


        if y.size != self.lms.n_neurons:
            #self.get_logger().warn("Taille de y incorrecte")
            return

        y = y.flatten()


        #self.get_logger().info(f" [LMS] Inputs x: {x}")
        #self.get_logger().info(f" [LMS] Targets y: {y}")


        if x.size > 0 and y.size > 0:
            self.lms.learn(x, y)



        # Prédire la vision à partir du LiDAR
        if x.size == 0:
            #self.get_logger().warn("⚠️ Aucun input pour le LMS, prédiction annulée.")
            return


        predicted_y = self.lms.s(x)
  
        i = 3
        error = y - predicted_y
        #self.get_logger().info(f"[Debug] Cellule d=12 : prédite = {predicted_y[i]:.3f}, réelle = {y[i]:.3f}, erreur = {error[i]:.3f}")
 
        self.get_logger().info(f"[LMS] Prédiction effectuée. Sample Y[] = {predicted_y}")
        predicted_matrix = predicted_y.reshape((4, 4))  # Moyenne des prédictions
        '''if np.any(np.abs(self.lms.w_ij) > 10):  
            self.get_logger().warn(" Poids LMS instables, réinitialisation en cours...")
            self.lms = LMS(n_neurons=2, input_size=2)  # Réinitialisation LMS'''
        
        if not self.bvc_matrix_buffer:
            #self.get_logger().warn("Aucune matrice BVC disponible pour le matching.")
            return


        # ============================
        # ERREUR ENTRE BVCs
        # ============================
        error_matrix = np.abs(closest_matrix - predicted_matrix)
        mean_error = np.mean(error_matrix)

        #self.get_logger().info(f"Erreur moyenne entre BVC prédite et réelle : {mean_error:.4f}")
        save_error_to_log(self.get_clock().now().nanoseconds / 1e9, mean_error)
     
        
        # ============================
        # PUBLICATION DES BVCs
        

        msg = Float64MultiArray()
        msg.data = predicted_matrix.flatten().tolist()
        self.publisher_predicted_bvc.publish(msg)
        #self.get_logger().info(f"[PUBLISH predicted_bvc] {predicted_matrix}")
        self.save_lms()


    def predict_only(self, visual_frame):
        self.get_logger().info("[PREDICTION SEULE] predict_only() appelée")
        x = np.array([[delta_az, delta_h] for _, delta_az, delta_h in visual_frame]).flatten()
        if x.size == 0:
            #self.get_logger().warn("Données d'entrée vides pour prédiction LMS → prédiction annulée.")
            return
        x = self.preprocess_input(visual_frame)
        print(f"[LMS] Input x pour prédiction seule: {x}")

        predicted_y = self.lms.s(x)
        predicted_matrix = predicted_y.reshape(4, 4)
        #self.get_logger().info(f" [PREDICTION SEULE] LMS a prédit: {predicted_y[:4]}")
        msg_pred = Float64MultiArray()
        msg_pred.data = predicted_matrix.flatten().tolist()
        self.publisher_predicted_bvc.publish(msg_pred)




          


def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraLMS()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
