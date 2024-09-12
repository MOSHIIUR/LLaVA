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

# Set to store unique image paths
unique_images = set()

# Extract unique image paths from each sample
for sample in data:
    if 'image' in sample:
        image_name = sample['image']
        unique_images.add(image_name)

# Number of unique images
total_unique_images = len(unique_images)

# Print the number of unique images
print(f"Total number of unique images across all samples: {total_unique_images}")