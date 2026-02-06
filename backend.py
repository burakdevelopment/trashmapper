import os
import time
import math
import cv2
import yaml
import json
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from dataclasses import dataclass

#SABİTLER VE SINIFLAR
LABELS = {
    0: 'Aluminium foil', 1: 'Battery', 2: 'Blister pack', 3: 'Carded blister',
    4: 'Other plastic bottle', 5: 'Clear plastic bottle', 6: 'Glass bottle', 7: 'Plastic cap',
    8: 'Metal cap', 9: 'Broken glass', 10: 'Food Can', 11: 'Aerosol', 12: 'Drink can',
    13: 'Toilet tube', 14: 'Carton', 15: 'Egg carton', 16: 'Drink carton', 17: 'Corrugated carton',
    18: 'Meal carton', 19: 'Pizza box', 20: 'Paper cup', 21: 'Plastic cup', 22: 'Foam cup',
    23: 'Glass cup', 24: 'Other cup', 25: 'Food waste', 26: 'Glass jar', 27: 'Plastic lid',
    28: 'Metal lid', 29: 'Other plastic', 30: 'Magazine', 31: 'Tissues', 32: 'Wrapping paper',
    33: 'Normal paper', 34: 'Paper bag', 35: 'Plastified bag', 36: 'Plastic film', 37: 'Six pack rings',
    38: 'Garbage bag', 39: 'Plastic wrapper', 40: 'Carrier bag', 41: 'PP bag',
    42: 'Crisp packet', 43: 'Spread tub', 44: 'Tupperware', 45: 'Disposable food box',
    46: 'Foam food box', 47: 'Other plastic container', 48: 'Plastic glooves', 49: 'Plastic utensils',
    50: 'Pop tab', 51: 'Rope', 52: 'Scrap metal', 53: 'Shoe', 54: 'Squeezable tube',
    55: 'Plastic straw', 56: 'Paper straw', 57: 'Styrofoam piece', 58: 'Unlabeled litter', 59: 'Cigarette'
}

COLORS = np.random.uniform(0, 255, size=(len(LABELS), 3)).astype(np.uint8)

@dataclass
class Detection:
    id: int
    label_id: int
    label_name: str
    conf: float
    box: list 
    center: tuple 

class TrashEngine:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        
        
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.cfg['ai']['model_path'], providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        
        
        self.cap = None
        self.init_camera()
        
        
        self.is_running = False
        self.start_time = None
        self.session_dir = ""
        self.grid_size = int(self.cfg['mapping']['grid_size_meter'] / self.cfg['mapping']['cell_size_meter'])
        self.grid_center = self.grid_size // 2
        
        self.heat_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        
        self.pose = {'x': 0.0, 'y': 0.0, 'yaw': 0.0} 
        self.detected_objects_log = []
        
        
        self.track_history = {} 
        self.next_track_id = 0

    def init_camera(self):
        c = self.cfg['camera']
        if c['type'] == 'picamera2':
            try:
                from picamera2 import Picamera2
                self.picam2 = Picamera2()
                config = self.picam2.create_video_configuration(
                    main={"size": (c['width'], c['height']), "format": "RGB888"},
                    controls={"FrameRate": c['fps'], "AfMode": 2, "AfRange": 2} # Sürekli Otomatik Odaklama
                )
                self.picam2.configure(config)
                self.picam2.start()
                self.cam_type = 'pi'
                print("PiCamera2 Module 3 has been initialized.")
            except Exception as e:
                print(f"PiCamera2 error: {e}. we are switching back to OpenCV.")
                self.cam_type = 'cv'
                self.cap = cv2.VideoCapture(0)
        else:
            self.cam_type = 'cv'
            self.cap = cv2.VideoCapture(0)
            self.cap.set(3, c['width'])
            self.cap.set(4, c['height'])

    def get_frame(self):
        if self.cam_type == 'pi':
            frame = self.picam2.capture_array() 
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
        else:
            ret, frame = self.cap.read()
            if not ret: return np.zeros((720,1280,3), np.uint8)
            return frame

    def preprocess(self, img):
        shape = img.shape[:2]
        new_shape = self.cfg['ai']['input_size']
        r = min(new_shape/shape[0], new_shape/shape[1])
        new_unpad = int(round(shape[1]*r)), int(round(shape[0]*r))
        dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
        dw, dh = dw/2, dh/2
        
        im = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        im = cv2.copyMakeBorder(im, int(round(dh-0.1)), int(round(dh+0.1)), 
                                int(round(dw-0.1)), int(round(dw+0.1)), 
                                cv2.BORDER_CONSTANT, value=(114,114,114))
        im = im.astype(np.float32) / 255.0
        im = im.transpose(2, 0, 1)
        im = np.expand_dims(im, axis=0)
        return im, r, (dw, dh)

    def detect(self, frame):
        
        blob, ratio, (dw, dh) = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: blob})[0]
        
        pred = np.transpose(outputs[0], (1, 0))
        
        boxes = []
        confidences = []
        class_ids = []
        
        rows = pred.shape[0]
        w, h = frame.shape[1], frame.shape[0]

        for i in range(rows):
            classes_scores = pred[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.cfg['ai']['conf_thres']:
                class_id = np.argmax(classes_scores)
                x, y, w_b, h_b = pred[i][0], pred[i][1], pred[i][2], pred[i][3]
                
                
                x = (x - dw) / ratio
                y = (y - dh) / ratio
                w_b /= ratio
                h_b /= ratio
                
                x1 = int(x - w_b / 2)
                y1 = int(y - h_b / 2)
                x2 = int(x + w_b / 2)
                y2 = int(y + h_b / 2)
                
                boxes.append([x1, y1, w_b, h_b])
                confidences.append(float(max_score))
                class_ids.append(class_id)
                
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.cfg['ai']['conf_thres'], self.cfg['ai']['iou_thres'])
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                x1, y1, w_b, h_b = box
                cx = x1 + w_b // 2
                cy = y1 + h_b // 2
                
                self.next_track_id += 1
                results.append(Detection(
                    id=self.next_track_id,
                    label_id=class_ids[i],
                    label_name=LABELS.get(class_ids[i], "Unknown"),
                    conf=confidences[i],
                    box=[x1, y1, x1+w_b, y1+h_b],
                    center=(cx, cy)
                ))
        return results

    def update_map(self, detections, dt):

        v = self.cfg['mapping']['virtual_speed']        
        dy = v * dt
        self.pose['y'] += dy
        
        cell_m = self.cfg['mapping']['cell_size_meter']
        
        robot_gx = int(self.pose['x'] / cell_m) + self.grid_center
        robot_gy = int(self.pose['y'] / cell_m) + self.grid_center
        
        robot_gx = np.clip(robot_gx, 0, self.grid_size-1)
        robot_gy = np.clip(robot_gy, 0, self.grid_size-1)

        for det in detections:
            img_w = self.cfg['camera']['width']
            fov = self.cfg['mapping']['fov_horizontal']
            
            offset_x_ratio = (det.center[0] - (img_w / 2)) / img_w
            
            dist_m = 1.5 
            
            angle_rad = offset_x_ratio * math.radians(fov)
            
            waste_dx = math.tan(angle_rad) * dist_m
            waste_dy = dist_m 
            
            wgx = int((self.pose['x'] + waste_dx) / cell_m) + self.grid_center
            wgy = int((self.pose['y'] + waste_dy) / cell_m) + self.grid_center
            
            if 0 <= wgx < self.grid_size and 0 <= wgy < self.grid_size:
                self.heat_grid[wgy, wgx] += 1 
                
            
            self.detected_objects_log.append({
                "ts": time.time(),
                "class": det.label_name,
                "conf": det.conf,
                "map_x": float(self.pose['x'] + waste_dx),
                "map_y": float(self.pose['y'] + waste_dy)
            })

    def start_session(self):
        self.is_running = True
        self.start_time = time.time()
        session_name = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.cfg['system']['save_dir'], session_name)
        os.makedirs(self.session_dir, exist_ok=True)
        
        self.heat_grid.fill(0)
        self.pose = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        self.detected_objects_log = []

    def stop_session(self):
        self.is_running = False
        return self.generate_report()

    def generate_report(self):
        
        plt.figure(figsize=(10, 10))
        plt.imshow(self.heat_grid, cmap='hot', interpolation='nearest', origin='lower')
        plt.colorbar(label='Waste Density')
        plt.title("Waste Heat Map (Without GPS - Relative Location)")
        
        center = self.grid_center
        plt.plot(center, center, 'go', label='Beginning')
        
        end_y = int(self.pose['y'] / self.cfg['mapping']['cell_size_meter']) + center
        plt.arrow(center, center, 0, end_y-center, color='blue', head_width=2, label='Route')
        plt.legend()
        
        heatmap_path = os.path.join(self.session_dir, "heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()
        
        
        counts = {}
        for obj in self.detected_objects_log:
            c = obj['class']
            counts[c] = counts.get(c, 0) + 1
            
        
        plt.figure(figsize=(12, 6))
        if counts:
            plt.bar(counts.keys(), counts.values(), color='orange')
            plt.xticks(rotation=45, ha='right')
        plt.title("Tespit Edilen Atık Dağılımı")
        plt.tight_layout()
        hist_path = os.path.join(self.session_dir, "histogram.png")
        plt.savefig(hist_path)
        plt.close()
        
        
        suggestions = []
        total = sum(counts.values())
        if "Cigarette" in counts and counts["Cigarette"] > 5:
            suggestions.append("⚠️ High number of cigarette butts: Ashtrays should be added to the area.")
        if "Plastic bottle" in str(counts) or "Clear plastic bottle" in counts:
            suggestions.append("⚠️ High density of plastic bottles: The visibility of recycling bins should be increased.")
        if total > 50:
            suggestions.append("⚠️ Overall pollution levels are high: Cleaning frequency should be increased.")
        
        report = {
            "session_dir": self.session_dir,
            "total_objects": total,
            "counts": counts,
            "suggestions": suggestions,
            "heatmap_path": heatmap_path,
            "hist_path": hist_path
        }
        
        with open(os.path.join(self.session_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
            
        return report

    def cleanup(self):
        if self.cam_type == 'pi':
            self.picam2.stop()
        elif self.cap:
            self.cap.release()
