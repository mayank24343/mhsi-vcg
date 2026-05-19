import torch
from PIL import Image
from transformers import (
    DetrForObjectDetection,
    DetrImageProcessor,
)
from config import DETR_THRESHOLD, RAM_THRESHOLD, DEVICE
from ram.models import ram_plus
from ram import inference_ram, get_transform


COCO_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A',
    'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A',
    'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]


class ObjectDetector:
    """
    Runs DETR and RAM++ on an image and returns the
    intersection of their detected object class names.
    """

    def __init__(self):
        print("[ObjectDetector] Loading DETR...")
        self.detr_processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-50"
        )
        self.detr_model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50"
        ).cpu()
        self.detr_model.eval()

        """
        print("[ObjectDetector] Loading RAM++...")
        #need to install ram thru github 
        self.ram_transform = get_transform(image_size=384)
        self.ram_model = ram_plus(
        pretrained="https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth",
        image_size=384,
        vit="swin_b"
        ).to(DEVICE)
        self.ram_model.eval()
        """

        print("[ObjectDetector] DETR model loaded.")

    @torch.no_grad()
    def detect_detr(self, image: Image.Image) -> set:
        """
        Run DETR on image.
        Returns set of detected class name strings.
        """
        inputs = self.detr_processor(
            images=image,
            return_tensors="pt"
        )

        outputs = self.detr_model(**inputs)

        # post process to get boxes and labels
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.detr_processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=DETR_THRESHOLD
        )[0]
        print(results)

        detected = set()
        for label in results["labels"]:
            class_name = COCO_CLASSES[label.item()]
            if class_name != 'N/A':
                detected.add(class_name)

        print(f"[DETR] Detected: {detected}")
        return detected

    @torch.no_grad()
    def detect_ram(self, image: Image.Image) -> set:
        """
        Run RAM++ on image.
        Returns set of detected tag/class name strings.
        """

        image_tensor = self.ram_transform(image).unsqueeze(0).half().to(DEVICE)

        tags, _ = inference_ram(image_tensor, self.ram_model)

        # RAM returns pipe-separated tags sometimes
        # normalize both comma and pipe separation
        raw_tags = tags[0]

        detected = set()

        for tag in raw_tags.replace('|', ',').split(','):
            tag = tag.strip().lower()
            if tag:
                detected.add(tag)

        print(f"[RAM++] Detected: {detected}")

        return detected
    def detect(self, image: Image.Image) -> list:
        """
        Run both models and return intersection.
        Returns list of class name strings.
        """
        detr_classes = self.detect_detr(image)
        # ram_classes = self.detect_ram(image)

        # intersection — only objects both models agree on
        intersection = detr_classes #& ram_classes

        # fallback: if intersection is empty use DETR alone
        if not intersection:
            print("[ObjectDetector] Intersection empty, falling back to DETR only.")
            intersection = detr_classes

        result = list(intersection)
        print(f"[ObjectDetector] Final objects: {result}")
        return result