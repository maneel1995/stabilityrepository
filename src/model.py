from typing import Dict
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image
import matplotlib.pyplot as plt

# Pick the largest batch size that can fit on our GPUs.
# If doing CPU inference you might need to lower considerably (e.g. to 10).
BATCH_SIZE = 1024


class ImageClassifier:
    def __init__(self):
        model_path = hf_hub_download(
            repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt"
        )
        self.yolo_model = YOLO(model_path)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        # Text prompts
        self.prompts = [
            "a photo of a person wearing eyeglasses",
            "a photo of a person wearing sunglasses",
            "a photo of a person not wearing glasses",
        ]

    # BEGIN: Updated __call__ method
    def __call__(self, batch: Dict[str, np.ndarray]):
        ##### Can be made more efficient by using a batch of images instead of processing one by one.
        filtered = []
        truth_values = []
        image_buffer = [
            np.asarray(Image.open(io.BytesIO(img_dict["bytes"])).convert("RGB"))
            for img_dict in batch["image"]
        ]
        for img_dict in batch["image"]:
            image_buffer = np.asarray(
                Image.open(io.BytesIO(img_dict["bytes"])).convert("RGB")
            )
            results = self.yolo_model(image_buffer)
            img_filtered = []
            val = False
            for result in results:
                if (result.boxes.cls.flatten() == 0).any() == False:
                    continue
                x1, y1, x2, y2 = result.boxes.xyxy[0].tolist()
                width = x2 - x1
                height = y2 - y1
                if width >= 100 and height >= 100:
                    # Get the cropped image and send it to CLIP for classification.
                    cropped_image = image_buffer[int(y1) : int(y2), int(x1) : int(x2)]
                    pil_image = Image.fromarray(cropped_image)
                    inputs = self.clip_processor(
                        text=self.prompts,
                        images=pil_image,
                        return_tensors="pt",
                        padding=True,
                    )
                    outputs = self.clip_model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=1)
                    max_prob, max_index = probs.max(dim=1)
                    if (
                        max_index.item() == 0
                    ):  # Check if the highest probability corresponds to the first class
                        val = True
                        img_filtered.append(
                            {
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                                "width": width,
                                "height": height,
                                "confidence": max_prob.item(),  # Add the confidence value
                            }
                        )
            filtered.append(img_filtered)
            truth_values.append(val)
        batch["bounding_boxes"] = filtered
        batch["truth_values"] = truth_values
        return batch
