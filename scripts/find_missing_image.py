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

# # Load the JSON file
# with open('./playground/data/llava_v1_5_mix665k.json', 'r') as file:
#     data = json.load(file)
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

# Function to count all files in a directory
def count_files_in_directory(directory):
    file_count = 0
    for root, dirs, files in os.walk(directory):
        file_count += len(files)
    return file_count

# Total count of files across all directories
total_file_count = 0

# Count files in each directory
for directory in image_directories:
    if os.path.exists(directory):
        count = count_files_in_directory(directory)
        total_file_count += count
        print(f"Directory '{directory}' contains {count} files.")
    else:
        print(f"Directory '{directory}' does not exist or is not accessible.")

# Final total count
print(f"Total number of files across all directories: {total_file_count}")