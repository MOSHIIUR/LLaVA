import json
import os

# Define the base directories where the images might be stored
image_directories = [
    'playground/data/coco/train2017',
    'playground/data/gqa/images',
    'playground/data/ocr_vqa/images',
    'playground/data/textvqa/train_images',
    'playground/data/vg/VG_100K',
    'playground/data/vg/VG_100K_2'
]

# Function to check if an image exists in any of the directories
def image_exists(image_name):
    for directory in image_directories:
        image_path = os.path.join(directory, image_name)
        if os.path.exists(image_path):
            return True
    return False

# Load the JSON file
with open('./playground/data/llava_v1_5_mix665k.json', 'r') as file:
    data = json.load(file)

# Initialize counters
total_samples = len(data)
missing_images_count = 0
processed_count = 0

# Iterate through each sample and check if the image exists
for sample in data:
    processed_count += 1
    image_name = sample['image']
    
    if not image_exists(image_name):
        print(f"Missing image: {image_name}")
        missing_images_count += 1

    # Print status every 100 samples processed
    if processed_count % 100 == 0:
        print(f"Processed {processed_count}/{total_samples} samples. Missing images so far: {missing_images_count}")

# Final status
print(f"Processing completed. Total samples: {total_samples}, Missing images: {missing_images_count}")
