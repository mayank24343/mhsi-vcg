# large vision language model used
LVLM_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct" #"openbmb/MiniCPM-V-2"#"llava-hf/llava-1.5-phi-2-hf"#"Qwen/Qwen2-VL-2B-Instruct"

# guidance hyperparameters
ALPHA = 1    # amplification strength
DETR_THRESHOLD = 0.5       # confidence threshold for DETR
RAM_THRESHOLD = 0.68         # confidence threshold for RAM++
TOP_SVD_COMPONENTS = 3   # None = keep all, int = keep top-k

# generation
MAX_NEW_TOKENS = 500

# device
DEVICE = "cuda"