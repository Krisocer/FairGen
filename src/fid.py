import os
import subprocess
import shutil
import argparse
from PIL import Image
import random

def resize_and_pad_image(input_path, output_path, size=(256, 256)):
    """ Read image, resize it, and save to target location """
    image = Image.open(input_path)
    # If image mode is RGBA, convert to RGB
    # if image.mode == 'RGBA':
    image = image.convert('RGB')
    image = image.resize(size, Image.LANCZOS)
    image.save(output_path)

def ensure_matching_image_counts_and_size(base_dir1, base_dir2, temp_dir1, temp_dir2, pairs):
    """ 
    Ensure temp_dir1 copies all images from specified categories in base_dir1, and temp_dir2 contains only randomly selected images of the same count
    """

    # Clear and recreate temporary folders
    if os.path.exists(temp_dir1):
        shutil.rmtree(temp_dir1)
    if os.path.exists(temp_dir2):
        shutil.rmtree(temp_dir2)

    os.makedirs(temp_dir1, exist_ok=True)
    os.makedirs(temp_dir2, exist_ok=True)

    total_base1_files = []  # Store all images from base_dir1
    total_base2_files = []  # Store randomly sampled images from base_dir2

    # Iterate through categories in pairs
    for category1, category2 in pairs:
        dir1 = os.path.join(base_dir1, category1)
        dir2 = os.path.join(base_dir2, category2)

        # Ensure directories exist
        if not os.path.exists(dir1) or not os.path.exists(dir2):
            print(f"⚠ Warning: {dir1} or {dir2} does not exist. Skipping this category.")
            continue

        # Get all images from base_dir1
        files1 = [os.path.join(dir1, f) for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))]
        files2 = [os.path.join(dir2, f) for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))]

        count1 = len(files1)
        count2 = len(files2)

        print(f"📂 Processing category: {category1} vs {category2}")
        print(f"  base_dir1: {count1} images, base_dir2: {count2} images")

        if count1 == 0:
            print(f"⚠ Warning: {dir1} has no images. Skipping this category.")
            continue

        if count2 < count1:
            print(f"⚠ Warning: {dir2} has only {count2} images, but base_dir1 needs {count1}. Skipping this category.")
            continue

        # Add images from base_dir1 directly to temp_dir1
        for file in files1:
            file_name = f"base1_{len(total_base1_files)}.jpg"  # Rename to avoid name conflicts across categories
            resize_and_pad_image(file, os.path.join(temp_dir1, file_name))
            total_base1_files.append(file_name)

        # Randomly select the same number of images from base_dir2 and add to temp_dir2
        selected_files = random.sample(files2, count1)
        for file in selected_files:
            file_name = f"base2_{len(total_base2_files)}.jpg"
            resize_and_pad_image(file, os.path.join(temp_dir2, file_name))
            total_base2_files.append(file_name)

    print(f"\n✅ Total images in temp_dir1: {len(total_base1_files)}")
    print(f"✅ Total images in temp_dir2: {len(total_base2_files)}")

    return temp_dir1, temp_dir2

def calculate_fid_for_selected_pairs(base_dir1, base_dir2, output_file, temp_dir):
    """ Calculate FID only for specified categories in pairs, ensuring temp directory image counts match base_dir1 """

    '''
    pairs = [
        # ("African_people_allergic_contact_dermatitis", "African_people_allergic_contact_dermatitis"),
        # ("African_people_basal_cell_carcinoma", "African_people_basal_cell_carcinoma"),
        # ("African_people_lichen_planus", "African_people_lichen_planus"),
        # ("African_people_psoriasis", "African_people_psoriasis"),
        # ("African_people_squamous_cell_carcinoma", "African_people_squamous_cell_carcinoma"),
        # ("Asian_people_allergic_contact_dermatitis", "Asian_people_allergic_contact_dermatitis"),
        # ("Asian_people_basal_cell_carcinoma", "Asian_people_basal_cell_carcinoma"),
        # ("Asian_people_lichen_planus", "Asian_people_lichen_planus"),
        # ("Asian_people_psoriasis", "Asian_people_psoriasis"),
        # ("Asian_people_squamous_cell_carcinoma", "Asian_people_squamous_cell_carcinoma"),
        ("Caucasian_people_allergic_contact_dermatitis", "Caucasian_people_allergic_contact_dermatitis"),
        ("Caucasian_people_basal_cell_carcinoma", "Caucasian_people_basal_cell_carcinoma"),
        ("Caucasian_people_lichen_planus", "Caucasian_people_lichen_planus"),
        ("Caucasian_people_psoriasis", "Caucasian_people_psoriasis"),
        ("Caucasian_people_squamous_cell_carcinoma", "Caucasian_people_squamous_cell_carcinoma")
    ]
    '''

    '''
    pairs = [
        # ("Demented_Age_Above_75", "Demented_Age_Above_75"),
        ("Demented_Age_Below_75", "Demented_Age_Below_75"),
        #("Nondemented_Age_Above_75", "Nondemented_Age_Above_75"),
        ("Nondemented_Age_Below_75", "Nondemented_Age_Below_75"),
    ]
    '''


    pairs = [
        ("female_COVID19", "female_Atelectasis"),
        ("female_Edema", "female_Cardiomegaly"),
        ("female_Lung_Opacity", "female_Consolidation"),
        ("female_No_Finding", "female_Edema"),
        ("female_Pleural_Effusion", "female_Pleural_Effusion")
        # ("male_COVID19", "male_Atelectasis"),
        # ("male_Edema", "male_Cardiomegaly"),
        # ("male_Lung_Opacity", "male_Consolidation"),
        # ("male_No_Finding", "male_Edema"),
        # ("male_Pleural_Effusion", "male_Pleural_Effusion")
    ]


    temp_dir1 = os.path.join(temp_dir, "temp_combined_1")
    temp_dir2 = os.path.join(temp_dir, "temp_combined_2")

    temp_dir1, temp_dir2 = ensure_matching_image_counts_and_size(base_dir1, base_dir2, temp_dir1, temp_dir2, pairs)

    # Calculate FID
    print("\n🚀 Calculating FID...")
    command = ["python", "-m", "pytorch_fid", temp_dir1, temp_dir2]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    fid_result = result.stdout.strip()

    with open(output_file, 'w') as f:
        if "FID:" in fid_result:
            fid_value = fid_result.split('FID:')[-1].strip()
            f.write(f"FID: {fid_value}\n")
            print(f"✅ Calculation complete. FID: {fid_value}")
        else:
            f.write("⚠ FID calculation failed or no output\n")
            print("⚠ FID calculation failed or no output")

    # Do not delete temporary directories for debugging purposes
    # shutil.rmtree(temp_dir1)
    # shutil.rmtree(temp_dir2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate FID for selected directory pairs.")
    parser.add_argument("base_dir2", type=str, help="Path to the second base directory")
    args = parser.parse_args()

    base_dir1 = "/ocean/projects/ccr200024p/zli27/sd_xray/input/downstream_xray"
    base_dir2 = args.base_dir2
    output_file = "/ocean/projects/ccr200024p/zli27/sd_xray/sd/src/fid_result/fid_results_sd.txt"
    temp_dir = "/ocean/projects/ccr200024p/zli27/sd_xray/sd/src/temp"

    calculate_fid_for_selected_pairs(base_dir1, base_dir2, output_file, temp_dir)
