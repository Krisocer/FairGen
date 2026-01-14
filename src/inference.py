import argparse
import os
import torch
import math
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

# =========================================================================
# 1. Configuration: Categories for each modality
# =========================================================================
CATEGORIES = {
    "skin": [
        f"{race}_people_{disease}" 
        for race in ["African", "Asian", "Caucasian"]
        for disease in ["allergic_contact_dermatitis", "basal_cell_carcinoma", "lichen_planus", "psoriasis", "squamous_cell_carcinoma"]
    ],
    "mri": [
        "Demented_Age_Above_75",
        "Demented_Age_Below_75",
        "Nondemented_Age_Above_75",
        "Nondemented_Age_Below_75"
    ],
    "chest": [
        f"{gender}_{disease}"
        for gender in ["male", "female"]
        for disease in ["COVID19", "Edema", "Lung_Opacity", "Normal", "Pleural_Effusion"]
    ]
}

def get_prompt(modality, category):
    """
    Generate the text prompt based on modality and category name.
    Ensures prompts match the training logic.
    """
    if modality == "skin":
        # e.g., "African_people_psoriasis" -> "African people psoriasis"
        return category.replace("_", " ")
    
    elif modality == "chest":
        # e.g., "male_Lung_Opacity" -> "male Lung Opacity"
        return category.replace("_", " ")
    
    elif modality == "mri":
        # e.g., "Demented_Age_Above_75" -> "Age Above 75 Demented"
        try:
            condition, age_group = category.split("_Age_") # Split into "Demented" and "Above_75"
            age_text = age_group.replace("_", " ") # "Above 75"
            return f"Age {age_text} {condition}"
        except ValueError:
            return category.replace("_", " ") # Fallback
            
    return category

def dummy_safety_checker(images, clip_input):
    return images, [False] * len(images)

def parse_args():
    parser = argparse.ArgumentParser(description="Universal Inference Script for FairGen")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned UNet folder (e.g., .../checkpoint-15000/unet)")
    parser.add_argument("--base_model", type=str, default="CompVis/stable-diffusion-v1-4", help="Base Stable Diffusion model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images")
    parser.add_argument("--modality", type=str, required=True, choices=["skin", "mri", "chest"], help="Target modality")
    
    parser.add_argument("--num_images_per_class", type=int, default=1000, help="Number of images to generate per class")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation (increase for speed if VRAM allows)")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Model
    print(f"Loading UNet from: {args.model_path}")
    unet = UNet2DConditionModel.from_pretrained(args.model_path, torch_dtype=torch.float16)
    
    print(f"Loading Pipeline based on: {args.base_model}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        unet=unet,
        torch_dtype=torch.float16,
        safety_checker=None # Disable safety checker to save memory/avoid false positives
    )
    
    # Explicitly disable safety checker if it loaded defaults
    if pipe.safety_checker is not None:
        pipe.safety_checker = dummy_safety_checker
        
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True) # Disable pipe's internal bar to keep logs clean

    # 2. Prepare Categories
    target_categories = CATEGORIES[args.modality]
    print(f"Generating for modality: {args.modality}")
    print(f"Categories: {target_categories}")

    # 3. Generation Loop
    torch.manual_seed(args.seed)
    
    for category in target_categories:
        prompt = get_prompt(args.modality, category)
        group_dir = os.path.join(args.output_dir, category)
        os.makedirs(group_dir, exist_ok=True)
        
        print(f"\nProcessing: {category}")
        print(f"Prompt: '{prompt}'")
        print(f"Target: {args.num_images_per_class} images")

        # Determine how many images are already done (Resume capability)
        existing_imgs = [f for f in os.listdir(group_dir) if f.endswith('.png')]
        current_count = len(existing_imgs)
        
        if current_count >= args.num_images_per_class:
            print(f"Skipping {category}, already has {current_count} images.")
            continue

        images_to_generate = args.num_images_per_class - current_count
        num_batches = math.ceil(images_to_generate / args.batch_size)
        
        generated_so_far = current_count
        
        for i in range(num_batches):
            # Calculate actual batch size for the last batch
            current_batch_size = min(args.batch_size, args.num_images_per_class - generated_so_far)
            
            try:
                # Generate
                outputs = pipe(
                    [prompt] * current_batch_size, 
                    num_inference_steps=args.steps, 
                    guidance_scale=args.guidance_scale
                )
                
                # Save
                for img in outputs.images:
                    save_path = os.path.join(group_dir, f"{category}_{generated_so_far+1:05d}.png")
                    img.save(save_path)
                    generated_so_far += 1
                    
                print(f"  Progress: {generated_so_far}/{args.num_images_per_class}", end='\r')
                
            except Exception as e:
                print(f"\nError generating batch for {category}: {e}")
                continue

    print(f"\n\nAll generation tasks for {args.modality} completed.")

if __name__ == "__main__":
    main()