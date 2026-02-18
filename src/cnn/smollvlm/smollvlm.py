#This is a simple example of using SmolVLM to capture an image 
# from the webcam, process it with a custom prompt, and 
# extract structured JSON output focused on food items.

# Unfortunately, after testing and fixing the script many times, I found that 
# SmolVLM's image understanding is very basic 
# and it often fails to identify food items accurately.
    # The generated JSON is often empty or contains irrelevant objects.
# This is likely because SmolVLM is a general-purpose model and not fine-tuned for food recognition.
# SmolVLM is used more for visual reasoning & description tasks, rather than specific object recognition like food classification.
# For better results, I will use YOLO as a food detection language together with a specialized model fine-tuned on food datasets (like Food101) from PyTorch's torchvision library. This combination should provide more accurate food identification and structured output.

#https://docs.pytorch.org/vision/0.19/generated/torchvision.datasets.Food101.html

#In the beginning, I have also used Llava from Lama which is great 
# for general image understanding and reasoning, but it struggled with specific food recognition. SmolVLM is similar in that regard, so I switched to a more specialized approach with YOLO and a fine-tuned ResNet18 model on Food101 for better accuracy in identifying food items.


import cv2
import torch
import json
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

device = torch.device("cpu")

# -----------------------------
# Load SmolVLM
# -----------------------------
model_id = "HuggingFaceTB/SmolVLM-Base"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id).to(device)

# -----------------------------
# Capture Image from Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

print("Press SPACE to capture image...")

while True:
    ret, frame = cap.read()
    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1)

    if key == 32:  # SPACE
        image_path = "captured.jpg"
        cv2.imwrite(image_path, frame)
        break

cap.release()
cv2.destroyAllWindows()

# -----------------------------
# Load Captured Image
# -----------------------------
image = Image.open(image_path).convert("RGB")

# -----------------------------
# Proper Chat Messages
# -----------------------------
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": """
You MUST respond with ONLY valid JSON.

Do NOT write 'User:' or 'Assistant:'.
Do NOT explain anything.
Do NOT add extra text.

Only include FOOD items (ignore people, furniture, objects).

Return JSON exactly in this format:

{
  "objects": ["food1", "food2"],
  "description": "short food-related description"
}
"""
            }
        ],
    }
]

# -----------------------------
# Apply Chat Template
# -----------------------------
text = processor.apply_chat_template(messages, add_generation_prompt=True)

inputs = processor(
    text=text,
    images=image,
    return_tensors="pt"
).to(device)

# -----------------------------
# Generate Output
# -----------------------------
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False
    )  

raw_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n--- RAW MODEL OUTPUT ---")
print(raw_output)

# -----------------------------
# Extract JSON Safely
# -----------------------------
try:
    json_start = raw_output.find("{")
    json_end = raw_output.rfind("}") + 1
    structured_output = json.loads(raw_output[json_start:json_end])
except:
    structured_output = {
        "objects": [],
        "description": "Model did not return valid JSON."
    }

print("\n--- FINAL STRUCTURED JSON (FOOD ONLY) ---")
print(json.dumps(structured_output, indent=4))














