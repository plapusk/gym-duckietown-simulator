import os
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import time
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from ultralytics import YOLO
import sys
import os
sys.path.append(os.path.abspath("../mask-train/Fast-SCNN-pytorch"))

from models.fast_scnn import get_fast_scnn

class PerceptionModule:
    def __init__(self, yolov8_path, deeplab_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo = YOLO(yolov8_path)
        self.seg_model = deeplabv3_resnet50(pretrained=False)
        self.seg_model.classifier[4] = nn.Conv2d(256, 4, kernel_size=1)
        self.seg_model.load_state_dict(torch.load(deeplab_path, map_location=self.device))
        self.seg_model.to(self.device).eval()
        self.seg_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, frame):
        start_time = time.time()

        results = self.yolo.predict(frame, conf=0.4, verbose=False)
        boxes = results[0].boxes

        # Get bounding box coordinates, confidence scores, and class IDs
        xyxy = boxes.xyxy.cpu().numpy()           # (N, 4)
        scores = boxes.conf.cpu().numpy()         # (N,)
        class_ids = boxes.cls.cpu().numpy()       # (N,)

        # Combine into a list of full bbox entries
        bboxes = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            score = scores[i]
            class_id = class_ids[i]
            bboxes.append((x1, y1, x2, y2, class_id, score))

        with torch.no_grad():
            input_tensor = self.seg_tf(frame).unsqueeze(0).to(self.device)
            output = self.seg_model(input_tensor)['out']
            seg_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

        elapsed = time.time() - start_time
        return bboxes, seg_mask, elapsed

class PerceptionModuleSCNN:
    def __init__(self, yolov8_path, scnn_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load YOLO model
        self.yolo = YOLO(yolov8_path)
        
        # Load Fast-SCNN model
        self.seg_model = get_fast_scnn(dataset='citys', pretrained=True, map_cpu=False)
        
        # Replace classifier head to 4 classes
        self.seg_model.classifier = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 4, kernel_size=1)  # 4 classes
        )
        
        # Load your trained weights
        self.seg_model.load_state_dict(torch.load(scnn_path, map_location=self.device))
        self.seg_model.to(self.device).eval()
        
        # Transform for SCNN input
        self.seg_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
    def predict(self, frame):
        start_time = time.time()
        
        # YOLO detection
        results = self.yolo.predict(frame, conf=0.4, verbose=False)
        bboxes = results[0].boxes.xyxy.cpu().numpy()
        
        # Segmentation with SCNN
        with torch.no_grad():
            input_tensor = self.seg_tf(frame).unsqueeze(0).to(self.device)
            output = self.seg_model(input_tensor)[0]  # take the first element if tuple
            seg_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
        
        elapsed = time.time() - start_time
        return bboxes, seg_mask, elapsed

import numpy as np
from scipy.ndimage import label, center_of_mass

def extract_state(bboxes, seg_mask, speed=None):
    state = []
    total_pixels = seg_mask.size

    # Lane classes: 2 = side-lane, 3 = mid-lane
    for lane_class in [2, 3]:
        mask = (seg_mask == lane_class)
        class_area = np.sum(mask) / total_pixels
        state.append(class_area)

        if lane_class == 2:
            # Side-lane: poate avea 2 laturi
            labeled_mask, num_features = label(mask)
            centroids = center_of_mass(mask, labeled_mask, range(1, num_features + 1))

            # Sortăm centroids după coordonata x (coloană)
            centroids = sorted(centroids, key=lambda c: c[1])

            # Adăugăm până la 2 centroiduri (stânga, dreapta)
            for i in range(2):
                if i < len(centroids):
                    state.extend(centroids[i])  # y, x
                else:
                    state.extend([0.0, 0.0])  # fallback
        else:
            # Mid-lane: presupunem un singur corp
            if np.sum(mask) > 0:
                coords = np.column_stack(np.where(mask))
                centroid = coords.mean(axis=0)
                state.extend(centroid.tolist())
            else:
                state.extend([0.0, 0.0])

    # Număr de obstacole
    state.append(len(bboxes))

    # Viteza
    if speed is not None:
        state.append(speed.get('linear', 0.0))
        state.append(speed.get('angular', 0.0))
    else:
        state.extend([0.0, 0.0])

    return np.array(state, dtype=np.float32)





def load_random_image_from_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    assert len(image_files) > 0, "Folderul nu conține imagini!"
    random_image = random.choice(image_files)
    image_path = os.path.join(folder_path, random_image)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, image_rgb, image_path


def draw_button(image, text="Next", position=(10, 50), size=(100, 40), color=(50, 50, 255)):
    x, y = position
    w, h = size
    cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
    cv2.putText(image, text, (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return (x, y, x + w, y + h)  # return bounding box of the button


def visualize_result(frame_bgr, bboxes, seg_mask, duration):
    for box in bboxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)

    color_map = np.array([
        [0, 0, 0],
        [0, 255, 255],
        [0, 255, 0],
        [255, 0, 255]
    ])
    seg_overlay = color_map[seg_mask]
    seg_overlay = cv2.addWeighted(frame_bgr, 0.6, seg_overlay.astype(np.uint8), 0.4, 0)

    cv2.putText(seg_overlay, f"Eval time: {duration:.3f} sec", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    button_coords = draw_button(seg_overlay, "Next", position=(10, 60))
    return seg_overlay, button_coords


# Variabilă globală pentru click
button_clicked = False


def mouse_callback(event, x, y, flags, param):
    global button_clicked  
    button_coords = param
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1, x2, y2 = button_coords
        if x1 <= x <= x2 and y1 <= y <= y2:
            button_clicked = True





if __name__ == "__main__":
    yolov8_path = "slow_color.pt"
    deeplab_path = "fine_tune_myset.pth"
    image_folder = "../dataset/images/"

    model = PerceptionModule(yolov8_path, deeplab_path)

    window_name = "Perception Output"
    cv2.namedWindow(window_name)

    while True:
        img_bgr, img_rgb, path = load_random_image_from_folder(image_folder)
        print(f"Imagine testată: {path}")

        bboxes, seg_mask, duration = model.predict(img_rgb)
        vis_img, button_coords = visualize_result(img_bgr.copy(), bboxes, seg_mask, duration)

        # Setează funcția de click cu coordonatele butonului
        cv2.setMouseCallback(window_name, mouse_callback, param=button_coords)

        button_clicked = False

        while True:
            cv2.imshow(window_name, vis_img)
            key = cv2.waitKey(20)

            if button_clicked or key == ord('q'):
                break

        if key == ord('q'):
            break

    cv2.destroyAllWindows()
