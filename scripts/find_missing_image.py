import json
import os
from PIL import Image

# Define the base directories where the images might be stored
directory = './playground/data'

# Function to check if an image exists in any of the directories
def image_exists(image_name):
    image_path = os.path.join(directory, image_name)
    if os.path.exists(image_path):
            return image_path
    return None

def is_image_valid(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify the image is not corrupted
        return True
    except (IOError, SyntaxError) as e:
        print(f"Invalid image file: {image_path}")
        return False

# Load the JSON file
with open('./playground/data/llava_v1_5_mix665k.json', 'r') as file:
    data = json.load(file)

# Initialize counters
total_samples = len(data)
missing_images_count = 0
missing_image_key_count = 0
processed_count = 0

# Iterate through each sample and check if the image exists and is valid
for sample in data:
    processed_count += 1
    
    # Check if the 'image' key exists in the sample
    if 'image' not in sample:
        missing_image_key_count += 1
        continue

    image_name = sample['image']
    image_path = image_exists(image_name)
    
    if image_path is None:
        print(f"Missing image: {image_name}")
        missing_images_count += 1
    elif not is_image_valid(image_path):
        print(f"Corrupted image: {image_name}")
        missing_images_count += 1

    # Print status every 100 samples processed
    if processed_count % 100 == 0:
        print(f"Processed {processed_count}/{total_samples} samples. Missing or corrupted images so far: {missing_images_count}")

# Final status
print(f"Processing completed. Total samples: {total_samples}, Missing images: {missing_images_count}, Missing 'image' key samples: {missing_image_key_count}")
