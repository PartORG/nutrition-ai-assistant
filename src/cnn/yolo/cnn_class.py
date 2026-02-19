# general imports
import cv2
import torch
import json
from ultralytics import YOLO
from torchvision import models, transforms, datasets
from PIL import Image

# building a class to encapsulate the entire pipeline for better modularity and reusability
class CNNPipeline:
    def __init__(
        self,
        yolo_path="src/cnn/models_cnn/yolov8n.pt",
        food_model_path="src/cnn/models_cnn/food101_resnet18_best.pth",
        data_root="./data_cnn",
        conf_threshold=0.6
    ):
        # device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # load YOLO model
        self.yolo_model = YOLO(yolo_path)

        # Load Food101 classifier and class names
        self.food_model = models.resnet18(weights=None)
        self.food_model.fc = torch.nn.Linear(
            self.food_model.fc.in_features, 101
        )
        self.food_model.load_state_dict(
            torch.load(food_model_path, map_location=self.device, weights_only=True)
        )
        self.food_model.to(self.device)
        self.food_model.eval()

        # Load class names
        food_dataset = datasets.Food101(
            root=data_root, split="train", download=False
        )
        self.class_names = food_dataset.classes

        # Transform for classifier
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # Filters (FIX)
        
        allowed_food_classes = {
            "banana", "apple", "orange", "pizza",
            "cake", "donut", "hot dog", "sandwich",
            "broccoli", "carrot"
        }
        self.FOOD_OBJECTS = {
            name for name in self.yolo_model.names.values()
            if name in allowed_food_classes
        }
        self.CONF_THRESHOLD = conf_threshold

    
    # Capture Image (from webcam, can be extended to load from file if needed)

    def capture_image(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise RuntimeError("Cannot access webcam")

        print("Press SPACE to capture image, ESC to quit...")

        while True:
            ret, frame = cap.read()
            cv2.imshow("Webcam", frame)

            key = cv2.waitKey(1)

            if key == 32:
                break
            elif key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                raise SystemExit("Cancelled.")
        cap.release()
        cv2.destroyAllWindows()

        return frame


    # Classify Crop (using the Food101 classifier)

    def classify_crop(self, crop):
        crop_pil = Image.fromarray(
            cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        )

        img_tensor = self.transform(crop_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.food_model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probs, 1).item()
            confidence = probs[0][predicted_idx].item()

        return self.class_names[predicted_idx], confidence

    # Run Detection + Classification on an Image
   
    def run(self, image_path=None):
        if image_path is None:
            frame = self.capture_image()
        else:
            frame = cv2.imread(image_path)

        results = self.yolo_model(frame)
        r = results[0]

        final_objects = []

        for box in r.boxes:
            label = r.names[int(box.cls[0])]
            conf = float(box.conf[0])

            if conf < self.CONF_THRESHOLD:
                continue

            if label not in self.FOOD_OBJECTS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            food_label, food_conf = self.classify_crop(crop)

            final_objects.append({
                "detected_object": label,
                "detection_confidence": round(conf, 3),
                "classified_food": food_label,
                "food_confidence": round(food_conf, 3)
            })

        return {
            "objects": final_objects,
            "description": f"Detected {len(final_objects)} FOOD-like objects in the image."
        }

# Optional standalone run for testing the pipeline without needing to import the class elsewhere

if __name__ == "__main__":
    pipeline = CNNPipeline()
    result = pipeline.run()

    print("\n--- FINAL PIPELINE OUTPUT ---")
    print(json.dumps(result, indent=4))










