import json
import os
import urllib.request

# --- Configuration ---
JSON_FILE_PATH = 'C:/Users/Mayank/OneDrive/Desktop/MHSI/data/pope/coco_pope_random.json'
SAVE_DIRECTORY = 'downloaded_images'
COCO_BASE_URL = 'http://images.cocodataset.org/val2014/'
DOWNLOAD_LIMIT = 500

def get_unique_images(file_path):
    """Reads the JSON Lines file and extracts unique image names."""
    unique_images = set()
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue # Skip empty lines
                
            try:
                data = json.loads(line)
                if 'image' in data:
                    unique_images.add(data['image'])
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON on line {line_number}")
                
    return list(unique_images)

def download_images(image_list, save_dir, base_url, limit):
    """Downloads images from the provided list up to the specified limit."""
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Slice the list to enforce the 500 image limit
    images_to_download = image_list[:limit]
    total = len(images_to_download)
    
    print(f"Found {len(image_list)} unique images. Preparing to download {total}...")
    
    for index, img_name in enumerate(images_to_download, 1):
        url = base_url + img_name
        save_path = os.path.join(save_dir, img_name)
        
        # Skip if the file already exists (saves time if you restart the script)
        if os.path.exists(save_path):
            print(f"[{index}/{total}] Already exists, skipping: {img_name}")
            continue
            
        print(f"[{index}/{total}] Downloading {img_name}...")
        try:
            urllib.request.urlretrieve(url, save_path)
        except Exception as e:
            print(f"Failed to download {img_name}. Error: {e}")

if __name__ == "__main__":
    # 1. Extract unique image names
    images = get_unique_images(JSON_FILE_PATH)
    
    # 2. Download the images
    if images:
        download_images(images, SAVE_DIRECTORY, COCO_BASE_URL, DOWNLOAD_LIMIT)
        print("\nDownload process complete!")
    else:
        print("No image filenames were found in the provided file.")