import io
from typing import Dict

import numpy as np
import pyarrow as pa
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO

class ImageClassifier:
    def __init__(self):
        model_path = hf_hub_download(
            repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt"
        )
        self.yolo_model = YOLO(model_path)
        self.yolo_model.training = False
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        # Text prompts
        self.promptsv1 = [
            "This is a human being or person",
            "This is not a human being or person",
        ]
        self.promptsv2 = [
            "a person wearing clear eyeglasses",
            "a person wearing black sunglasses",
            "a photo of a person not wearing glasses",
        ]

    # BEGIN: Updated __call__ method
    def __call__(self, batch: Dict[str, np.ndarray]):
        ##### Can be made more efficient by using a batch of images instead of processing one by one.
        bbox_struct = pa.struct(
            [
                ("bounding_box", pa.bool_()),
                ("x1", pa.float32()),
                ("y1", pa.float32()),
                ("x2", pa.float32()),
                ("y2", pa.float32()),
                ("width", pa.float32()),
                ("height", pa.float32()),
                ("confidence", pa.float32()),
                ("box_confidence", pa.float32()),
                ("clip_person_confidence", pa.float32()),
                ("clip_eyeglass_confidence", pa.float32()),
            ]
        )
        filtered = [[{"bounding_boxes": False}] for x in range(len(batch["image"]))]
        truth_values = []
        for idx,img_dict in enumerate(batch["image"]):
            image_buffer = np.asarray(
            Image.open(io.BytesIO(img_dict["bytes"])).convert("RGB")
        )
            results = self.yolo_model(image_buffer)
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
                inputs = self.clip_processor(
                    text = self.promptsv1,
                    images = pil_image,
                    return_tensors="pt",
                    padding=True,
                )
                outputs = self.clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)
                max_prob, max_index = probs.max(dim=1)
                if max_index.item() == 1:
                    continue
                clip_person_confidence = max_prob.item()
                inputs = self.clip_processor(
                    text=self.promptsv2,
                    images=pil_image,
                    return_tensors="pt",
                    padding=True,
                )
                outputs = self.clip_model(**inputs)
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
                        }
                    )
            if len(img_filtered) > 0:
                filtered[idx] = img_filtered
            truth_values.append(val)
        array_arrow = pa.array(filtered, type=pa.list_(bbox_struct))
        batch["bounding_boxes"] = array_arrow
        batch["truth_values"] = truth_values
        return batch
