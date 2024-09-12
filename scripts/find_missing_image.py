import json
import os

# Define the base directories where the images might be stored
directory = './playground/data'

# Function to check if an image exists in any of the directories
def image_exists(image_name):
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
missing_image_key_count = 0
processed_count = 0

# Iterate through each sample and check if the image exists
for sample in data:
    processed_count += 1
    
    # Check if the 'image' key exists in the sample
    if 'image' not in sample:
        # print(f"Missing 'image' key in sample with ID: {sample.get('id', 'Unknown ID')}")
        # missing_image_key_count += 1
        continue

    image_name = sample['image']
    
    if not image_exists(image_name):
        print(f"Missing image: {image_name}")
        missing_images_count += 1

    # Print status every 100 samples processed
    if processed_count % 100 == 0:
        print(f"Processed {processed_count}/{total_samples} samples. Missing images so far: {missing_images_count}")

# Final status
print(f"Processing completed. Total samples: {total_samples}, Missing images: {missing_images_count}")