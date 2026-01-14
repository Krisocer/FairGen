import os
import subprocess
import shutil
import argparse
from PIL import Image
import random

def resize_and_pad_image(input_path, output_path, size=(256, 256)):
    image = Image.open(input_path)

    # If image mode is RGBA, convert to RGB
    # if image.mode == 'RGBA':
    image = image.convert('RGB')
    image = image.resize(size, Image.LANCZOS)
    image.save(output_path)

def prepare_images_for_comparison(base_dir1, base_dir2, temp_dir1, temp_dir2):
    # Ensure temp directories are empty
    if os.path.exists(temp_dir1):
        shutil.rmtree(temp_dir1)
    if os.path.exists(temp_dir2):
        shutil.rmtree(temp_dir2)
    
    os.makedirs(temp_dir1, exist_ok=True)
    os.makedirs(temp_dir2, exist_ok=True)

    # Count the number of images in each category in base_dir1
    category_counts = {}  # Store count of each category in base_dir1
    base1_files = []  # Store all image paths from base_dir1
    
    for category in os.listdir(base_dir1):
        category_path = os.path.join(base_dir1, category)
        if os.path.isdir(category_path):
            files = [os.path.join(category_path, f) for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
            category_counts[category] = len(files)
            base1_files.extend(files)

    print("\nImage count per category in base_dir1:")
    for category, count in category_counts.items():
        print(f"{category}: {count}")

    # Copy all images from base_dir1 to temp_dir1
    print("\nCopying base_dir1 images to temp_dir1...")
    for idx, file in enumerate(base1_files):
        file_name = f"base1_{idx}.jpg"
        resize_and_pad_image(file, os.path.join(temp_dir1, file_name))
    
    # Copy randomly selected images from base_dir2 to temp_dir2
    print("\nRandomly selecting images from base_dir2 to temp_dir2...")
    base2_files = []
    
    for category, count in category_counts.items():
        category_path = os.path.join(base_dir2, category)
        if os.path.exists(category_path) and os.path.isdir(category_path):
            files = [os.path.join(category_path, f) for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
            if len(files) < count:
                print(f"⚠ Warning: base_dir2 {category} has only {len(files)} images, but base_dir1 needs {count}. Skipping this category.")
                continue
            selected_files = random.sample(files, count)  # Randomly select count images
            base2_files.extend(selected_files)
    
    # Copy selected images from base_dir2 to temp_dir2
    for idx, file in enumerate(base2_files):
        file_name = f"base2_{idx}.jpg"
        resize_and_pad_image(file, os.path.join(temp_dir2, file_name))
    
    print(f"\nFinal Temp_dir1 has {len(os.listdir(temp_dir1))} images")
    print(f"Final Temp_dir2 has {len(os.listdir(temp_dir2))} images")

    return temp_dir1, temp_dir2

def calculate_fid(base_dir1, base_dir2, output_file, temp_dir):
    # Set temporary folder paths
    temp_dir1 = os.path.join(temp_dir, "temp_combined_1")
    temp_dir2 = os.path.join(temp_dir, "temp_combined_2")

    temp_dir1, temp_dir2 = prepare_images_for_comparison(base_dir1, base_dir2, temp_dir1, temp_dir2)

    print("\nCalculating FID...")
    command = ["python", "-m", "pytorch_fid", temp_dir1, temp_dir2]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    fid_result = result.stdout.strip()

    with open(output_file, 'w') as f:
        if "FID:" in fid_result:
            fid_value = fid_result.split('FID:')[-1].strip()
            f.write(f"FID for combined images: {fid_value}\n")
            print(f"✅ FID for combined images: {fid_value}")
        else:
            f.write(f"⚠ FID for combined images: Error or No Output\n")
            print(f"⚠ FID for combined images: Error or No Output")

    # Do not delete temporary files for debugging purposes
    shutil.rmtree(temp_dir1)
    shutil.rmtree(temp_dir2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate FID for combined directories.")
    parser.add_argument("base_dir2", type=str, help="Path to the second base directory")
    args = parser.parse_args()

    base_dir1 = "/ocean/projects/ccr200024p/zli27/sd_xray/input/downstream_xray"
    base_dir2 = args.base_dir2
    temp_dir = "/ocean/projects/ccr200024p/zli27/sd_xray/sd/src/temp"
    output_file = "/ocean/projects/ccr200024p/zli27/sd_xray/sd/src/fid_result/fidall_results_sd.txt"

    calculate_fid(base_dir1, base_dir2, output_file, temp_dir)
