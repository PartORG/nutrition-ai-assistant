# General imports

import cv2
import torch
import json
from ultralytics import YOLO
from torchvision import models, transforms, datasets
from PIL import Image
from collections import Counter     # for counting food classes in description and demoving duplicates in final JSON output


# 1. Device

device = torch.device("cpu")


# 2. Load YOLO

yolo_model = YOLO("src/cnn/models_cnn/yolov8n.pt") 


# 3. Load Food101 Classifier

food_model = models.resnet18(weights=None)
food_model.fc = torch.nn.Linear(food_model.fc.in_features, 101)
food_model.load_state_dict(torch.load("src/cnn/models_cnn/food101_model.pth", map_location=device, weights_only=True))
food_model.to(device)
food_model.eval()

# 4. Load class names
food_dataset = datasets.Food101(root="./data_cnn", split="train", download=False)
class_names = food_dataset.classes

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# 5. Filters (FIX)

FOOD_OBJECTS = {
    "banana", "apple", "orange", "pizza", "cake", "donut", "hot dog", "sandwich"
}
CONF_THRESHOLD = 0.5  # in order to skip low-confidence detections


# 6. Capture Image

cap = cv2.VideoCapture(0)

print("Press SPACE to capture image...")

while True:
    ret, frame = cap.read()
    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1)

    if key == 32:
        image_path = "captured.jpg"
        cv2.imwrite(image_path, frame)
        break
    elif key == 27:  # ESC
        cap.release()
        cv2.destroyAllWindows()
        raise SystemExit("Cancelled.")

cap.release()
cv2.destroyAllWindows()


# 7. YOLO Detection

results = yolo_model(image_path)
r = results[0]

final_objects = []

for i, box in enumerate(r.boxes):
    label = r.names[int(box.cls[0])]
    conf = float(box.conf[0])

    # confidence filter
    if conf < CONF_THRESHOLD:
        continue

    # only classify food-like detections
    if label not in FOOD_OBJECTS:
        # If only to keep non-food objects in JSON with null food classification:
        # final_objects.append({
        #     "detected_object": label,
        #     "detection_confidence": round(conf, 3),
        #     "classified_food": None
        # })
        continue

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # safety clamp (avoid out-of-bounds)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        continue
  

    # 8. Classification (NO DISK WRITE)

    # Convert OpenCV BGR → RGB → PIL
    crop_pil = Image.fromarray(
        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    )

    img_tensor = transform(crop_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = food_model(img_tensor)
        # Get predicted class index
        predicted_idx = torch.argmax(outputs, 1).item()      
        #Convert logits → probabilities
        probs = torch.softmax(outputs, dim=1)
        # Confidence of predicted class
        confidence = probs[0][predicted_idx].item()

    food_label = class_names[predicted_idx]

    final_objects.append({
        "detected_object": label,
        "detection_confidence": round(conf, 3),
        "classified_food": food_label,
        "food_confidence": round(confidence, 3)   
    })

# 9. Final JSON Output

output = { "objects": final_objects, "description": f"Detected {len(final_objects)} FOOD-like objects in the image." }

print("\n--- FINAL PIPELINE OUTPUT ---")
print(json.dumps(output, indent=4))










