import numpy as np
import rclpy
from rclpy.node import Node
import pickle
from std_msgs.msg import Float64MultiArray, Float32MultiArray ,Float32
import sys
from NL_LMS import LMS
import os
import datetime


#ros2 param set /lidar_camera_lms eval_only true


log_dir = os.path.expanduser("~/ros_logs")  
os.makedirs(log_dir, exist_ok=True)  
log_file = os.path.join(log_dir, "lidar_camera_errors.log") 
weights_file = os.path.join(log_dir, "lms_model.pkl")

train_trace_file = os.path.join(log_dir, "lms_train_trace.csv")
predict_trace_file = os.path.join(log_dir, "lms_predict_trace.csv")

# entêtes (une seule fois)
for f, header in [
    (train_trace_file,  "ts,delta_t,mode,feat0,feat1,feat2,feat3,feat4,target_top_row,target_top_bin,pred_top_row,pred_top_bin,mae,mae_row0,mae_row1,mae_row2,mae_row3,w_min,w_max,w_norm"),
    (predict_trace_file,"ts,mode,feat0,feat1,feat2,feat3,feat4,pred_top_row,pred_top_bin,pred_row0_bin,pred_row1_bin,pred_row2_bin,pred_row3_bin,w_min,w_max,w_norm"),
]:
    if not os.path.exists(f):
        with open(f, "w") as fh: fh.write(header + "\n")

eval_mats_file = os.path.join(log_dir, "lms_eval_matrices.csv")
if not os.path.exists(eval_mats_file):
    with open(eval_mats_file, "w") as fh:
        hdr_feats = ",".join([f"feat{i}" for i in range(5)])
        hdr_real16 = ",".join([f"real_{i}" for i in range(16)])
        hdr_pred16 = ",".join([f"pred_{i}" for i in range(16)])
        fh.write(f"ts,{hdr_feats},{hdr_real16},{hdr_pred16}\n")




class LidarCameraLMS(Node):
    def __init__(self):
        super().__init__('lidar_camera_lms')
        with open(log_file, "w") as f:
            f.write("timestamp,error_bvc\n") 
        
        self.lms = LMS(n_neurons=16, input_size=5) 
       
        if os.path.isfile(weights_file):
            try:
                with open(weights_file, 'rb') as f:
                    self.lms = pickle.load(f)
                    print("LMS model loaded successfully.")
                self.get_logger().info(f"LMS chargé depuis {weights_file}")
            except Exception as e:
                self.get_logger().warn(f"Erreur chargement LMS : {e}")
        #self.subscription_camera = self.create_subscription(Float64MultiArray, "/visualSystem/delta_yx", self.camera_callback, 10)
        self.subscription_features = self.create_subscription(Float64MultiArray, "/visualSystem/distance_features", self.features_callback, 10)

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
        self.declare_parameter('eval_only', False)
        self.eval_only = self.get_parameter('eval_only').get_parameter_value().bool_value
        
        self.declare_parameter('predict_with_targets', False)
        self.predict_with_targets = self.get_parameter('predict_with_targets').get_parameter_value().bool_value
        
        self.publisher_predicted_vision = self.create_publisher(Float64MultiArray, 'predicted_vision', 10)
        self.publisher_predicted_bvc = self.create_publisher(Float64MultiArray, 'predicted_bvc', 10)
        #stockage des moyennes pour les deltas
        self.moyennes_delta_az_list = []
        self.moyennes_delta_h_list = []
        self.delta_h_vs_distance = [] 
        self.camera_timestamps = []
        self.train_count = 0 
        self.current_features = None

        self.lidar_data = []  
        self.camera_data = []  
        self.all_predictions = []
        self.vehicle_heading = None
        self.last_lidar_ts = None
        self.bvc_matrix_buffer = [] 

    '''def camera_callback(self, msg):
        data = msg.data
        timestamp = data[0]
        mean_delta_y = data[1]
        mean_delta_x = data[2]
  

        self.camera_data.append((timestamp, [mean_delta_x, mean_delta_y]))
        self.camera_data = self.camera_data[-50:]  # garder les dernières 50 frames

        if self.lidar_data:
            self.train_lms(timestamp, mean_delta_x, mean_delta_y)'''
    
    def features_callback(self, msg):
        self.current_features = np.array(msg.data, dtype=float)
        self.features_ts = float(self.current_features[0])
        self.current_features = self.current_features[1:]   
        '''if self.get_parameter('eval_only').get_parameter_value().bool_value:
            self.predict_only()'''

            
    def activity_matrix_callback(self, msg):
        try: 
            data = msg.data
            timestamp = data[0]
            matrix = np.array(data[1:]).reshape((4, 4))

            self.bvc_matrix_buffer.append((timestamp, matrix))
            self.bvc_matrix_buffer = self.bvc_matrix_buffer[-300:]  
            self.last_bvc_ts = self.get_clock().now().nanoseconds / 1e9  
            #self.get_logger().info(f" Matrice BVC reçue @ {timestamp:.2f}s")
            if getattr(self, "features_ts", None) is not None:
                self.train_lms(self.features_ts) 
        except Exception as e:
            self.get_logger().warning(f"Format de matrice invalide reçu : {e}.")
          

    def lidar_callback(self, msg):
        self.last_lidar_ts = self.get_clock().now().nanoseconds
        data = msg.data
        if len(data) < 3:
            return  

        timestamp = data[0]
        lidar_entries = list(zip(data[1::2], data[2::2]))

        lidar_angles = np.radians([az for az, _ in lidar_entries]) % (2 * np.pi)
        lidar_entries_radians = [(np.radians(az) % (2 * np.pi), dist) for az, dist in lidar_entries]

        self.lidar_data.append((timestamp, lidar_entries_radians))
        self.lidar_data = self.lidar_data[-100:]
    
    def _row_argmaxes(self, mat):
        idx = np.argmax(mat, axis=1)
        val = mat[np.arange(mat.shape[0]), idx]
        return idx.tolist(), val.tolist()

    def _w_stats(self):
        w = getattr(self.lms, "w_ij", None)
        if w is None: 
            return (0.0, 0.0, 0.0)
        return float(np.min(w)), float(np.max(w)), float(np.linalg.norm(w))
    
    def _top_row_and_bin(self, mat):
        row_max = mat.max(axis=1)
        top_row = int(np.argmax(row_max))
        top_bin = int(np.argmax(mat[top_row]))
        return top_row, top_bin

    def soft_bin_encoding(self, bin_index, num_bins=4, sigma=0.5):
        values = np.arange(num_bins)
        weights = np.exp(-0.5 * ((values - bin_index) / sigma) ** 2)
        return weights / np.sum(weights)
    
    def heading_callback(self, msg):
        self.vehicle_heading = np.radians(msg.data) % (2 * np.pi)

    def preprocess_input(self, mean_dx, mean_dy): 
        #scale = 0.1 
        return np.array([float(mean_dx), float(mean_dy)], dtype=float)
  
    def save_error_to_log(self, timestamp, error):
        with open(log_file, "a") as f:
            f.write(f"{timestamp},{error}\n")

    def soft_one_hot(self, k, num_bins=4, sigma=0.7):
        i = np.arange(num_bins)
        d = np.minimum(np.abs(i-k), num_bins-np.abs(i-k))
        w = np.exp(-0.5*(d/sigma)**2)
        return w/(w.sum()+1e-12)


    def train_lms(self, cam_ts):
        '''if not self.bvc_matrix_buffer:
            self.get_logger().warn("Pas de BVC disponible → prédiction seule")
            self.predict_only()
            return'''
        self.eval_only = self.get_parameter('eval_only').get_parameter_value().bool_value
        timestamps = np.array([ts for ts, _ in self.bvc_matrix_buffer])
        time_diffs = np.abs(timestamps - cam_ts)
        closest_idx = np.argmin(time_diffs)
        delta_t = time_diffs[closest_idx]

        '''if delta_t > 0.1:
            self.get_logger().warn(f"Δt trop grand ({delta_t:.3f}s) → prédiction seule")
            self.predict_only()
            return'''

        closest_matrix = self.bvc_matrix_buffer[closest_idx][1]
        self.get_logger().info(f"[TRAIN] Δt={delta_t:.3f}s, cam_ts={cam_ts:.3f}")

        #x = self.preprocess_input(mean_dx, mean_dy)
        if self.current_features is None:
            self.get_logger().warn("Pas encore de features distance reçues")
            return

        x = self.current_features
        if x.size == 0 or closest_matrix.size != self.lms.n_neurons:
            return
        
        # Filtrage des directions inactives pour éviter d'apprendre sur du "vide"
        eps = 1e-6
        y_matrix = closest_matrix.copy()

        # ignore les lignes non informatives
        y_matrix[np.max(y_matrix, axis=1) < 1e-6] = 0.0
        t_idx, t_val = self._row_argmaxes(y_matrix)
        self.get_logger().debug(f"[TRAIN] Target top bins per row (N,E,S,O) = {t_idx} | vals={['%.3f'%v for v in t_val]}")


        # lissage léger entre bins + normalisation par ligne
        '''k = np.array([0.25, 0.5, 0.25])
        for i in range(4):
            row = y_matrix[i]
            if row.sum() > 0:
                row = 0.25*np.roll(row, -1) + 0.5*row + 0.25*np.roll(row, 1)
                s = row.sum()
                y_matrix[i] = row / (s + eps)'''
        if float(y_matrix.max()) < 1e-6 or float(y_matrix.sum()) == 0.0:
            self.get_logger().debug("Skip train: empty/flat target")
            return
        
        # normalisation par ligne
        # filtrage
 
        y_matrix[np.max(y_matrix, axis=1) < 1e-6] = 0.0
        
        active_row = int(np.argmax(y_matrix.max(axis=1)))
        for r in range(4):
            if r != active_row:
                y_matrix[r, :] = 0.0

        k = int(np.argmax(y_matrix[active_row]))
        y_matrix[active_row] = self.soft_one_hot(k, num_bins=4, sigma=0.7)

        sigma = 0.75   
        alpha = 1.0   


        if not self.eval_only:
            self.lms.learn(x, y_matrix.flatten())
            self.get_logger().debug(f"[TRAIN] x=[{x[0]:.4f},{x[1]:.4f}]")
            self.train_count += 1
            self.get_logger().info(f" LMS entraîné {self.train_count} fois")
        else:
            self.get_logger().info("[EVAL] Pas d'update des poids (comparaison seulement)")

        predicted_y = self.lms.s(x)
        predicted_matrix = predicted_y.reshape((4, 4))
        t_row, t_bin = self._top_row_and_bin(y_matrix)
        p_row, p_bin = self._top_row_and_bin(predicted_matrix)


        p_idx, p_val = self._row_argmaxes(predicted_matrix)

        error_matrix = np.abs(y_matrix - predicted_matrix)
        mae = float(np.mean(error_matrix))
        mae_rows = np.mean(error_matrix, axis=1)
        w_min, w_max, w_norm = self._w_stats()

        if self.eval_only:
            try:
                with open(eval_mats_file, "a") as fh:
                    ts_now = float(self.get_clock().now().nanoseconds/1e9)
                    feats = ",".join(f"{v:.6f}" for v in x.tolist())
                    real_flat = y_matrix.flatten().astype(float).tolist()
                    pred_flat = predicted_matrix.flatten().astype(float).tolist()
                    fh.write(f"{ts_now:.6f},{feats},")
                    fh.write(",".join(f"{v:.6f}" for v in real_flat) + ",")
                    fh.write(",".join(f"{v:.6f}" for v in pred_flat) + "\n")
            except Exception as e:
                    self.get_logger().error(f"Erreur écriture {eval_mats_file}: {e}")

        self.get_logger().info(
            f"[TRAIN] MAE={mae:.4f} | rows={['%.4f'%m for m in mae_rows]} | pred top bins={p_idx}"
        )

        mode_str = "eval" if self.eval_only else "train"
        with open(train_trace_file, "a") as fh:
            ts_now = float(self.get_clock().now().nanoseconds/1e9)
            feats = ",".join(f"{v:.6f}" for v in x.tolist())  # x = features (5)
            fh.write(
                f"{ts_now:.6f},{float(delta_t):.6f},{'eval' if self.eval_only else 'train'},"
                f"{feats},"
                f"{t_row},{t_bin},{p_row},{p_bin},"
                f"{mae:.6f},{float(mae_rows[0]):.6f},{float(mae_rows[1]):.6f},"
                f"{float(mae_rows[2]):.6f},{float(mae_rows[3]):.6f},"
                f"{w_min:.6f},{w_max:.6f},{w_norm:.6f}\n"
            )

        # DEBUG: Affichage des plages d’entrée, sortie et cible
        print("Entrée LMS:", x)
        print("Cible BVC (apprise) max/min:", y_matrix.max(), y_matrix.min())
        print("Sortie LMS max/min:", predicted_y.max(), predicted_y.min())

        msg = Float64MultiArray()
        msg.data = predicted_matrix.flatten().tolist()
        self.publisher_predicted_bvc.publish(msg)
        error_matrix = np.abs(y_matrix - predicted_matrix)

        mean_error = np.mean(error_matrix)
        self.save_error_to_log(self.get_clock().now().nanoseconds / 1e9, mean_error)
        try:
            with open(weights_file, 'wb') as f:
                pickle.dump(self.lms, f)
            self.get_logger().info(" LMS sauvegardé dans lms_model.pkl")
        except Exception as e:
            self.get_logger().error(f" Erreur sauvegarde LMS : {e}")

    '''def predict_only(self):
        if self.current_features is None:
            self.get_logger().warn("Aucune feature distance dispo")
            return
        x = self.current_features
        print(f"[Predict] Entrée LMS : {x}")
        print(f"[Predict] Poids max: {np.max(self.lms.w_ij):.4f} | min: {np.min(self.lms.w_ij):.4f}")
        if x.size == 0:
            return

        predicted_y = self.lms.s(x)
      
        predicted_matrix = predicted_y.reshape((4, 4))
        p_idx, _ = self._row_argmaxes(predicted_matrix)
        w_min, w_max, w_norm = self._w_stats()
        p_row, p_bin = self._top_row_and_bin(predicted_matrix)


        self.get_logger().info(f"[PRED] x=[{x[0]:.4f},{x[1]:.4f}] | pred top bins={p_idx} | w:[{w_min:.4f},{w_max:.4f}] norm={w_norm:.3f}")

        with open(predict_trace_file, "a") as fh:
            fh.write("{ts:.6f},predict,{dx:.6f},{dy:.6f},{pr},{pb},{pbrow0},{pbrow1},{pbrow2},{pbrow3},{wmin:.6f},{wmax:.6f},{wnorm:.6f}\n".format(
                ts=float(self.get_clock().now().nanoseconds/1e9),
                dx=float(x[0]), dy=float(x[1]),
                pr=p_row, pb=p_bin,
                pbrow0=int(np.argmax(predicted_matrix[0])),
                pbrow1=int(np.argmax(predicted_matrix[1])),
                pbrow2=int(np.argmax(predicted_matrix[2])),
                pbrow3=int(np.argmax(predicted_matrix[3])),
                wmin=w_min, wmax=w_max, wnorm=w_norm
        ))

        print(f"[Predict] Prédiction LMS : max={np.max(predicted_y):.4f}, min={np.min(predicted_y):.4f}")
        print("Prédiction LMS:", predicted_matrix)
    
        msg_pred = Float64MultiArray()
        msg_pred.data = predicted_matrix.flatten().tolist()
        self.publisher_predicted_bvc.publish(msg_pred)'''



def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraLMS()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


