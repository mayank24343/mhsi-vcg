import fiftyone as fo
import fiftyone.zoo as foz

#
# Load 50 random samples from the validation split
#
# Only the required images will be downloaded (if necessary).
# By default, only detections are loaded
#

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    max_samples=500,
    shuffle=True,
)

session = fo.launch_app(dataset)

# see fiftyone website for more