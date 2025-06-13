## This script downloads data from Hugging Face and checks for human presence in images using YOLOv8 and CLIP.
from typing import Dict
import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO

model_path = hf_hub_download(
    repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt"
)
yolo_model = YOLO(model_path)
yolo_model.training = False
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# Text prompts
promptsv1 = [
    "This is a human being or person",
    "This is not a human being or person",
]
promptsv2 = [
    "a person wearing clear eyeglasses",
    "a person wearing black sunglasses",
    "a person not wearing eyeglasses or sunglasses",
]
def check_now(image_path):
    image_buffer = np.asarray(
        Image.open(image_path).convert("RGB")
    )
    results = yolo_model(image_buffer)
    img_filtered = []
    val = False
    for result in results:
                if (result.boxes.cls.flatten() == 0).any() == False or (result.boxes.cls[0] !=0).any():
                    continue
                x1, y1, x2, y2 = result.boxes.xyxy[0].tolist()
                width = x2 - x1
                height = y2 - y1
                box_confidence = result.boxes.conf[0].item()
                if width*height < 10000:
                    continue
                    # Get the cropped image and send it to CLIP for classification.
                cropped_image = image_buffer[int(y1) : int(y2), int(x1) : int(x2)]
                pil_image = Image.fromarray(cropped_image)
                inputs = clip_processor(
                    text = promptsv1,
                    images = pil_image,
                    return_tensors="pt",
                    padding=True,
                )
                outputs = clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)
                max_prob, max_index = probs.max(dim=1)
                print(probs)
                if max_index.item() == 1:
                    continue
                clip_person_confidence = max_prob.item()
                inputs = clip_processor(
                    text=promptsv2,
                    images=pil_image,
                    return_tensors="pt",
                    padding=True,
                )
                outputs = clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)
                max_prob, max_index = probs.max(dim=1)
                if (
                    max_index.item() == 0
                ):
                    val = True
                    img_filtered.append(
                        {
                            "bounding_box": True,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "width": width,
                            "height": height,
                            "confidence": max_prob.item(),  # Add the confidence value
                            "box_confidence": box_confidence,
                            "clip_person_confidence": clip_person_confidence,
                            "clip_eyeglass_confidence": max_prob.item(),
                        })
    return {"bounding_boxes": img_filtered, "valid": val}


check = check_now('saved_images/image_0.png')
print(check)

