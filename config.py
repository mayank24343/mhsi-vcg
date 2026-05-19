# large vision language model used
LVLM_MODEL_NAME = "llava-hf/llava-1.5-7b-hf"

# guidance hyperparameters
ALPHA = 1.0                  # amplification strength
DETR_THRESHOLD = 0.2       # confidence threshold for DETR
RAM_THRESHOLD = 0.68         # confidence threshold for RAM++
TOP_SVD_COMPONENTS = None    # None = keep all, int = keep top-k

# generation
MAX_NEW_TOKENS = 200

# device
DEVICE = "cuda"