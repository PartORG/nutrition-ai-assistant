# CNN Module – Vision Pipeline Documentation

# Overview

For the CNN-based computer vision component of this project, three different vision models were evaluated:

1. LLaVA (via Ollama)

2. SmolVLM (HuggingFace)

3. YOLO + Food101 fine-tuned classifier


The goal was to detect food items from images (including webcam input) and return structured JSON output suitable for backend integration.


# 1. Evaluation of Vision-Language Models - LLaVA

LLaVA is a general-purpose vision-language model capable of:

- Image understanding

- Visual reasoning

- Natural language description generation


While LLaVA performed well for general image descriptions, it struggled with:

- Accurate identification of specific food items

- Consistent structured JSON output

- Reliable object-level detection


It is strong for reasoning tasks but not optimized for specialized food recognition.


# 2. SmolVLM

SmolVLM is another general-purpose vision-language model.
After extensive testing and debugging:

- Image understanding was found to be basic

- Food detection accuracy was inconsistent

- JSON outputs were often empty or contained irrelevant objects

- It is more suited for visual reasoning and captioning tasks

SmolVLM is not fine-tuned for food recognition and therefore did not meet the requirements of this nutrition-focused AI assistant.


# 3. Final Architecture Decision: YOLO + Food101 Classifier

After evaluation, a more specialized computer vision pipeline was implemented:

🔹 YOLO (Object Detection)

- Model: yolov8n.pt

- Purpose: Detect objects in the image

- Output: Bounding boxes, labels, and confidence scores

YOLO is used for robust object detection in real time.


🔹 Food Classification (Transfer Learning)

- Base model: ResNet18

- Dataset: Food101 (from PyTorch torchvision)

- First model "food101_model.pth" was fine-tuned on a subset of the dataset (20000 data, 5 epochs, CPU-based training). 

- Final model "food101_resnet18_best.pth" was fine-tuned on the whole training set (GPU-based training via Google Colab, 15 epochs). The model was trained on the full Food101 training split (75,750 images, ~75% of the dataset) and evaluated on the official test split (25,250 images). 

- Output: Food class prediction (101 categories). 

- After a total of 15 epochs: Train Acc: 0.6767 | Test Acc: 0.7612

- Food101 documentation:
https://docs.pytorch.org/vision/0.19/generated/torchvision.datasets.Food101.html



# 4. Final Integrated Vision Pipeline Class

The final production-ready pipeline follows this structure:

- Webcam captures image

- YOLO detects objects

- Each detected object is cropped

For each crop:

- Classification is applied using the Food101 model

- Only applied to food-like YOLO classes

- Results are combined into structured JSON

JSON output can be:

- Printed

- Saved

- Sent to backend API


# 5. Structured JSON Output Format

This structured output ensures:

- Backend compatibility

- Future integration with nutrition database

- Calorie estimation extension



# 6. Architectural Justification

The final decision to use YOLO + Food101 was based on:

- Higher accuracy for food recognition

- Better control over detection and classification

- Clear separation between: Object detection and Food classification

- Structured, reliable JSON output

- Easier extensibility

This approach is more suitable for a nutrition-focused AI assistant than general-purpose vision-language models.


# Final Conclusion

While LLaVA and SmolVLM are powerful general-purpose multimodal models, they are not specialized for food recognition tasks.
The combination of:

- YOLO (object detection), ResNet18 fine-tuned on Food101 (classification) and Structured JSON output provides a robust, modular, and production-ready computer vision pipeline for this project.