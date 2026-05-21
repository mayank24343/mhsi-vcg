import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Set


# COCO synonyms maps common variants to canonical COCO category name
# This is the standard synonym list used in the original CHAIR paper
SYNONYMS = {
    "person":       ["person", "man", "woman", "child", "boy", "girl",
                     "people", "human", "kid", "baby", "lady", "guy",
                     "male", "female", "player", "rider", "skier",
                     "snowboarder", "surfer", "pedestrian", "crowd"],
    "bicycle":      ["bicycle", "bike", "cycle"],
    "car":          ["car", "vehicle", "automobile", "sedan", "suv",
                     "taxi", "cab"],
    "motorcycle":   ["motorcycle", "motorbike", "moped"],
    "airplane":     ["airplane", "plane", "aircraft", "jet", "airliner"],
    "bus":          ["bus", "coach"],
    "train":        ["train", "locomotive", "subway"],
    "truck":        ["truck", "lorry", "van", "pickup"],
    "boat":         ["boat", "ship", "vessel", "canoe", "kayak",
                     "sailboat", "ferry"],
    "traffic light":["traffic light", "stoplight", "traffic signal"],
    "fire hydrant": ["fire hydrant", "hydrant"],
    "stop sign":    ["stop sign"],
    "parking meter":["parking meter"],
    "bench":        ["bench", "seat"],
    "bird":         ["bird", "pigeon", "seagull", "duck", "owl",
                     "parrot", "penguin", "chicken", "hen", "eagle",
                     "hawk", "crow", "sparrow", "dove"],
    "cat":          ["cat", "kitten", "feline"],
    "dog":          ["dog", "puppy", "canine", "hound", "pup"],
    "horse":        ["horse", "pony", "stallion", "mare", "foal"],
    "sheep":        ["sheep", "lamb", "ram", "ewe"],
    "cow":          ["cow", "bull", "cattle", "calf", "ox", "buffalo",
                     "bison"],
    "elephant":     ["elephant"],
    "bear":         ["bear", "panda"],
    "zebra":        ["zebra"],
    "giraffe":      ["giraffe"],
    "backpack":     ["backpack", "bag", "rucksack", "knapsack",
                     "satchel", "pack"],
    "umbrella":     ["umbrella", "parasol"],
    "handbag":      ["handbag", "purse", "clutch"],
    "tie":          ["tie", "necktie"],
    "suitcase":     ["suitcase", "luggage", "briefcase", "bag"],
    "frisbee":      ["frisbee", "disc"],
    "skis":         ["skis", "ski"],
    "snowboard":    ["snowboard"],
    "sports ball":  ["ball", "football", "soccer ball", "basketball",
                     "baseball", "tennis ball", "volleyball"],
    "kite":         ["kite"],
    "baseball bat": ["baseball bat", "bat"],
    "baseball glove":["baseball glove", "glove", "mitt"],
    "skateboard":   ["skateboard", "board"],
    "surfboard":    ["surfboard", "board"],
    "tennis racket":["tennis racket", "racket", "racquet"],
    "bottle":       ["bottle", "jar", "container"],
    "wine glass":   ["wine glass", "glass", "goblet", "wineglass"],
    "cup":          ["cup", "mug", "glass"],
    "fork":         ["fork"],
    "knife":        ["knife", "blade"],
    "spoon":        ["spoon", "ladle"],
    "bowl":         ["bowl", "dish", "basin"],
    "banana":       ["banana"],
    "apple":        ["apple"],
    "sandwich":     ["sandwich", "burger", "sub", "panini"],
    "orange":       ["orange", "tangerine", "clementine"],
    "broccoli":     ["broccoli"],
    "carrot":       ["carrot"],
    "hot dog":      ["hot dog", "hotdog", "sausage", "frankfurter"],
    "pizza":        ["pizza", "pie"],
    "donut":        ["donut", "doughnut"],
    "cake":         ["cake", "pastry", "cupcake", "muffin"],
    "chair":        ["chair", "seat", "stool", "throne"],
    "couch":        ["couch", "sofa", "settee", "loveseat"],
    "potted plant": ["potted plant", "plant", "flower", "cactus",
                     "houseplant"],
    "bed":          ["bed", "mattress", "cot", "bunk"],
    "dining table": ["dining table", "table", "desk", "counter"],
    "toilet":       ["toilet", "commode", "lavatory", "loo"],
    "tv":           ["tv", "television", "monitor", "screen", "display",
                     "telly"],
    "laptop":       ["laptop", "computer", "notebook", "macbook"],
    "mouse":        ["mouse"],
    "remote":       ["remote", "remote control", "controller"],
    "keyboard":     ["keyboard"],
    "cell phone":   ["cell phone", "phone", "smartphone", "mobile",
                     "iphone", "android"],
    "microwave":    ["microwave", "microwave oven"],
    "oven":         ["oven", "stove", "range"],
    "toaster":      ["toaster"],
    "sink":         ["sink", "basin", "faucet"],
    "refrigerator": ["refrigerator", "fridge", "freezer"],
    "book":         ["book", "novel", "magazine", "newspaper",
                     "textbook"],
    "clock":        ["clock", "watch", "timer"],
    "vase":         ["vase", "pot", "urn"],
    "scissors":     ["scissors", "shears"],
    "teddy bear":   ["teddy bear", "teddy", "stuffed animal",
                     "plush", "toy"],
    "hair drier":   ["hair drier", "hair dryer", "dryer", "blowdryer"],
    "toothbrush":   ["toothbrush"],
}

# build reverse lookup: synonym → canonical name
SYNONYM_TO_CANONICAL: Dict[str, str] = {}
for canonical, synonyms in SYNONYMS.items():
    for syn in synonyms:
        SYNONYM_TO_CANONICAL[syn.lower()] = canonical


@dataclass
class CHAIRSample:
    """One image + caption sample for CHAIR evaluation."""
    image_id: str
    image_file: str
    ground_truth_objects: Set[str]    # canonical COCO names
    caption: str = ""                 # filled in after generation
    mentioned_objects: Set[str] = None
    hallucinated_objects: Set[str] = None

@dataclass
class CHAIRMetrics:
    chair_s: float
    chair_i: float
    recall: float
    total_captions: int
    total_sentences: int 
    total_mentioned: int
    total_hallucinated: int
    total_gt_objects: int

    def __str__(self):
        return (
            f"CHAIR_S:  {self.chair_s:.2f}%  "
            f"(sentences with hallucination)\n"
            f"CHAIR_I:  {self.chair_i:.2f}%  "
            f"(hallucinated / all mentioned)\n"
            f"Recall:   {self.recall:.2f}%  "
            f"(gt objects correctly mentioned)\n"
            f"---\n"
            f"Captions:            {self.total_captions}\n"
            f"Sentences:           {self.total_sentences}\n" 
            f"Mentioned objects:   {self.total_mentioned}\n"
            f"Hallucinated:        {self.total_hallucinated}\n"
            f"GT objects:          {self.total_gt_objects}"
        )


def extract_objects_from_caption(caption: str) -> Set[str]:
    """
    Parse a caption string and return the set of COCO canonical
    object names mentioned, using the synonym list.

    Args:
        caption: raw generated caption string

    Returns:
        set of canonical COCO category names mentioned
    """
    caption_lower = caption.lower()

    # tokenize — split on spaces and punctuation
    tokens = re.findall(r"[a-z]+", caption_lower)
    token_set = set(tokens)

    # also check bigrams and trigrams for multi-word categories
    words = caption_lower.split()
    bigrams  = [f"{words[i]} {words[i+1]}"
                for i in range(len(words) - 1)]
    trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}"
                for i in range(len(words) - 2)]

    all_ngrams = set(tokens) | set(bigrams) | set(trigrams)

    mentioned = set()
    for ngram in all_ngrams:
        if ngram in SYNONYM_TO_CANONICAL:
            mentioned.add(SYNONYM_TO_CANONICAL[ngram])

    return mentioned

def split_into_sentences(text: str) -> List[str]:
    """
    Split a caption into individual sentences.
    Handles common punctuation: . ! ?
    Filters out empty strings.
    """
    # split on . ! ? followed by whitespace or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # filter empty and very short fragments
    sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
    
    return sentences if sentences else [text.strip()]


def compute_chair_metrics(samples: List[CHAIRSample]) -> CHAIRMetrics:
    """
    Compute CHAIR_S, CHAIR_I and Recall over a list of samples.

    CHAIR_S: fraction of SENTENCES containing at least one hallucinated object
    CHAIR_I: fraction of all mentioned objects that are hallucinated
    Recall:  fraction of ground truth objects correctly mentioned
    """
    total_sentences = 0
    sentences_with_hallucination = 0

    total_mentioned = 0
    total_hallucinated = 0
    total_gt_objects = 0

    for sample in samples:
        gt = sample.ground_truth_objects

        # all objects mentioned anywhere in the full caption
        # used for CHAIR_I and Recall
        all_mentioned = extract_objects_from_caption(sample.caption)
        all_hallucinated = all_mentioned - gt

        sample.mentioned_objects = all_mentioned
        sample.hallucinated_objects = all_hallucinated

        # CHAIR_I numerator/denominator — caption level
        total_mentioned    += len(all_mentioned)
        total_hallucinated += len(all_hallucinated)
        total_gt_objects   += len(gt)

        # CHAIR_S — sentence level
        sentences = split_into_sentences(sample.caption)
        total_sentences += len(sentences)

        for sentence in sentences:
            mentioned_in_sentence = extract_objects_from_caption(sentence)
            hallucinated_in_sentence = mentioned_in_sentence - gt
            if hallucinated_in_sentence:
                sentences_with_hallucination += 1

    chair_s = (100.0 * sentences_with_hallucination / total_sentences
               if total_sentences > 0 else 0.0)

    chair_i = (100.0 * total_hallucinated / total_mentioned
               if total_mentioned > 0 else 0.0)

    recall  = (100.0 * (total_mentioned - total_hallucinated) / total_gt_objects
               if total_gt_objects > 0 else 0.0)

    return CHAIRMetrics(
    chair_s=chair_s,
    chair_i=chair_i,
    recall=recall,
    total_captions=len(samples),
    total_sentences=total_sentences,   
    total_mentioned=total_mentioned,
    total_hallucinated=total_hallucinated,
    total_gt_objects=total_gt_objects
    )


class CHAIRDataset:
    """
    Builds CHAIR evaluation samples from COCO annotations.

    Args:
        annotation_file: path to COCO instances_val2014.json
        num_images:      how many images to sample (paper uses 500)
        min_objects:     only use images with at least this many gt objects
        seed:            random seed
    """

    def __init__(
        self,
        annotation_file: str,
        num_images: int = 500,
        min_objects: int = 3,
        seed: int = 42
    ):
        import random
        self.num_images = num_images
        self.min_objects = min_objects
        random.seed(seed)
        self._random = random

        print("[CHAIR] Loading COCO annotations...")
        self.samples = self._build_samples(annotation_file)
        print(f"[CHAIR] Built {len(self.samples)} samples.")

    def _build_samples(self, annotation_file: str) -> List[CHAIRSample]:
        with open(annotation_file, 'r') as f:
            coco = json.load(f)

        # category id → name
        cat_id_to_name = {
            cat['id']: cat['name']
            for cat in coco['categories']
        }

        # image id → filename
        id_to_file = {
            img['id']: img['file_name']
            for img in coco['images']
        }

        # image id → set of canonical object names
        image_objects: Dict[int, Set[str]] = {}
        for ann in coco['annotations']:
            img_id = ann['image_id']
            name = cat_id_to_name[ann['category_id']]
            image_objects.setdefault(img_id, set()).add(name)

        # filter by min objects
        eligible = {
            img_id: objs
            for img_id, objs in image_objects.items()
            if len(objs) >= self.min_objects
        }

        # sample
        sampled_ids = self._random.sample(
            list(eligible.keys()),
            min(self.num_images, len(eligible))
        )

        return [
            CHAIRSample(
                image_id=str(img_id),
                image_file=id_to_file[img_id],
                ground_truth_objects=eligible[img_id]
            )
            for img_id in sampled_ids
        ]