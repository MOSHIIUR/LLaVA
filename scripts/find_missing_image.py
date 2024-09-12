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

# Set to store unique top-level directories
unique_directories = set()

# Extract and store unique top-level directories from each 'image' key
for sample in data:
    if 'image' in sample:
        image_path = sample['image']
        # Extract the top-level directory part of the image path
        top_level_directory = image_path.split('/')[0]
        unique_directories.add(top_level_directory)

# Print the unique top-level directories found
print("Unique top-level directories found in the JSON data:")
for directory in sorted(unique_directories):
    print(directory)