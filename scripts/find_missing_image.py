import json
import os

# # Define the base directories where the images might be stored
# image_directories = [
#     'playground/data'
# ]

# # Function to check if an image exists in any of the directories
# def image_exists(image_name):
#     for directory in image_directories:
#         image_path = os.path.join(directory, image_name)
#         if os.path.exists(image_path):
#             return True
#     return False

# Load the JSON file
with open('./playground/data/llava_v1_5_mix665k.json', 'r') as file:
    data = json.load(file)
# # Initialize counters and lists
# total_samples = len(data)
# missing_images_count = 0
# missing_image_key_count = 0
# processed_count = 0
# samples_without_image_key = []  # To store samples missing the 'image' key

# # Iterate through each sample and check if the image exists
# for sample in data:
#     processed_count += 1
    
#     # Check if the 'image' key exists in the sample
#     if 'image' not in sample:
#         print(f"Missing 'image' key in sample with ID: {sample.get('id', 'Unknown ID')}")
#         missing_image_key_count += 1
#         samples_without_image_key.append(sample)  # Store the sample
#         continue

#     image_name = sample['image']
    
#     if not image_exists(image_name):
#         print(f"Missing image: {image_name}")
#         missing_images_count += 1

#     # Print status every 100 samples processed
#     if processed_count % 100 == 0:
#         print(f"Processed {processed_count}/{total_samples} samples. Missing images so far: {missing_images_count}. Missing 'image' key samples: {missing_image_key_count}")

# # Write samples missing the 'image' key to a new JSON file
# with open('samples_missing_image_key.json', 'w') as outfile:
#     json.dump(samples_without_image_key, outfile, indent=4)

# # Final status
# print(f"Processing completed. Total samples: {total_samples}, Missing images: {missing_images_count}, Missing 'image' key samples: {missing_image_key_count}")
# print(f"Samples without 'image' key saved to 'samples_missing_image_key.json'")


# Define the base directories where the images are stored
image_directories = [
    './playground/data/coco/train2017',
    './playground/data/gqa/images',
    './playground/data/ocr_vqa/images',
    './playground/data/textvqa/train_images',
    './playground/data/vg/VG_100K',
    './playground/data/vg/VG_100K_2'
]

# Set to store unique file extensions
image_extensions = set()

# Extract the file extension from each 'image' key
for sample in data:
    if 'image' in sample:
        image_name = sample['image']
        _, ext = os.path.splitext(image_name)  # Split the file name and extension
        if ext:
            image_extensions.add(ext)  # Add the extension to the set (converted to lowercase)

# Print the unique image extensions found
print("Unique image file extensions found in the JSON data:")
print(image_extensions)
