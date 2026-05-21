import json
import random
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class POPEQuestion:
    image_id: str
    image_file: str
    object_name: str
    question: str
    answer: str           # "yes" or "no"
    sampling: str         # "random", "popular", "adversarial"


class POPEDataset:
    """
    Builds POPE evaluation questions from COCO annotations.

    Paper definition:
        For each image, sample l//2 positive objects (ground truth)
        and l//2 negative objects (not in image) using three strategies.

    Args:
        annotation_file: path to COCO instances_val2014.json
        num_images:      how many images to sample (paper uses 500)
        questions_per_image: l in the paper (paper uses 6)
        min_objects:     only use images with at least this many objects
        seed:            random seed for reproducibility
    """

    def __init__(
        self,
        annotation_file: str,
        num_images: int = 500,
        questions_per_image: int = 6,
        min_objects: int = 3,
        seed: int = 42
    ):
        self.num_images = num_images
        self.questions_per_image = questions_per_image
        self.min_objects = min_objects
        self.seed = seed
        random.seed(seed)

        print("[POPE] Loading COCO annotations...")
        self._load_annotations(annotation_file)
        print(f"[POPE] Loaded {len(self.image_objects)} images with objects.")

    def _load_annotations(self, annotation_file: str):
        """
        Parse COCO annotations to build:
            image_objects: {image_id -> set of object category names}
            image_files:   {image_id -> filename string}
            all_categories: list of all category names in the dataset
            category_frequencies: Counter of how often each category appears
        """
        with open(annotation_file, 'r') as f:
            coco = json.load(f)

        # build category id name lookup
        cat_id_to_name = {
            cat['id']: cat['name']
            for cat in coco['categories']
        }

        # build image id filename lookup
        self.image_files = {
            img['id']: img['file_name']
            for img in coco['images']
        }

        # build image id set of object names
        self.image_objects: Dict[int, set] = {}
        self.category_frequencies = Counter()

        for ann in coco['annotations']:
            img_id = ann['image_id']
            cat_name = cat_id_to_name[ann['category_id']]

            if img_id not in self.image_objects:
                self.image_objects[img_id] = set()

            self.image_objects[img_id].add(cat_name)
            self.category_frequencies[cat_name] += 1

        self.all_categories = list(cat_id_to_name.values())

        # filter to images with enough objects
        self.image_objects = {
            img_id: objs
            for img_id, objs in self.image_objects.items()
            if len(objs) >= self.min_objects
        }

    def _sample_images(self) -> List[int]:
        """Randomly sample num_images image IDs."""
        all_ids = list(self.image_objects.keys())
        #sampled = random.sample(all_ids, min(self.num_images, len(all_ids)))
        return all_ids

    def _make_question(self, obj: str) -> str:
        """Paper template: 'Is there a/an <object> in the image?'"""
        article = "an" if obj[0].lower() in "aeiou" else "a"
        return f"Is there {article} {obj} in the image?"

    def _sample_negatives_random(
        self,
        image_id: int,
        k: int
    ) -> List[str]:
        """Random sampling: randomly pick objects not in this image."""
        present = self.image_objects[image_id]
        candidates = [c for c in self.all_categories if c not in present]
        return random.sample(candidates, min(k, len(candidates)))

    def _sample_negatives_popular(
        self,
        image_id: int,
        k: int
    ) -> List[str]:
        """
        Popular sampling: top-k most frequent objects
        in the whole dataset that are not in this image.
        """
        present = self.image_objects[image_id]
        negatives = []
        for cat, _ in self.category_frequencies.most_common():
            if cat not in present:
                negatives.append(cat)
            if len(negatives) == k:
                break
        return negatives

    def _sample_negatives_adversarial(
        self,
        image_id: int,
        k: int
    ) -> List[str]:
        """
        Adversarial sampling: top-k objects that most frequently
        co-occur with the ground truth objects but are not in this image.

        Co-occurrence: how often category C appears in images that
        also contain any of the ground truth objects.
        """
        present = self.image_objects[image_id]
        co_occurrence = Counter()

        for other_id, other_objs in self.image_objects.items():
            if other_id == image_id:
                continue
            # if this image shares any object with the current image
            if other_objs & present:
                for obj in other_objs:
                    if obj not in present:
                        co_occurrence[obj] += 1

        negatives = []
        for cat, _ in co_occurrence.most_common():
            if cat not in present:
                negatives.append(cat)
            if len(negatives) == k:
                break

        # fallback if not enough co-occurring objects found
        if len(negatives) < k:
            negatives += self._sample_negatives_random(
                image_id, k - len(negatives)
            )

        return negatives[:k]

    def build(self, sampling: str) -> List[POPEQuestion]:
        """
        Build POPE questions for a given sampling strategy.

        Args:
            sampling: "random", "popular", or "adversarial"

        Returns:
            list of POPEQuestion objects
        """
        assert sampling in ("random", "popular", "adversarial"), \
            f"Unknown sampling strategy: {sampling}"

        sampled_ids = self._sample_images()
        k = self.questions_per_image // 2   # half positive, half negative

        questions = []
        for image_id in sampled_ids:
            present = list(self.image_objects[image_id])
            file_name = self.image_files[image_id]

            # positive objects, sample k from ground truth
            pos_objects = random.sample(present, min(k, len(present)))

            # negative objects, sample k using chosen strategy
            if sampling == "random":
                neg_objects = self._sample_negatives_random(image_id, k)
            elif sampling == "popular":
                neg_objects = self._sample_negatives_popular(image_id, k)
            else:
                neg_objects = self._sample_negatives_adversarial(image_id, k)

            # build positive questions
            for obj in pos_objects:
                questions.append(POPEQuestion(
                    image_id=str(image_id),
                    image_file=file_name,
                    object_name=obj,
                    question=self._make_question(obj),
                    answer="yes",
                    sampling=sampling
                ))

            # build negative questions
            for obj in neg_objects:
                questions.append(POPEQuestion(
                    image_id=str(image_id),
                    image_file=file_name,
                    object_name=obj,
                    question=self._make_question(obj),
                    answer="no",
                    sampling=sampling
                ))

        print(f"[POPE] Built {len(questions)} questions "
              f"({sampling} sampling, {len(sampled_ids)} images).")
        return questions