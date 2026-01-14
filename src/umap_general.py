import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import umap
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader



# which modalities？("skin", "mri", "xray")
MODALITY = "skin" 

if MODALITY == "skin":
    # real testing dataset
    REAL_TEST_SET_ROOT = "/ocean/projects/ccr200024p/zli27/sd_xray/output"
    # Vanilla 
    MODEL_PATH_VANILLA = "/ocean/projects/ccr200024p/zli27/sd_xray/sd/src/model_for_umap/skin_final_seed3_num800" 
    # FairGen model
    MODEL_PATH_FAIRGEN = "/ocean/projects/ccr200024p/zli27/sd_xray/sd/src/model_for_umap/skin_epoch5_seed3_num900" 
    
elif MODALITY == "mri":
    REAL_TEST_SET_ROOT = "/ocean/projects/ccr200024p/zli27/sd_xray/output" # Please verify path
    MODEL_PATH_VANILLA = "/ocean/projects/ccr200024p/zli27/sd_xray/sd/src/model_for_umap/mri_epoch5_seed3_num500"
    MODEL_PATH_FAIRGEN = "/ocean/projects/ccr200024p/zli27/sd_xray/sd/src/model_for_umap/mri_final_seed3_num500"

elif MODALITY == "xray":
    REAL_TEST_SET_ROOT = "/ocean/projects/ccr200024p/zli27/sd_xray/output" # Please verify path
    MODEL_PATH_VANILLA = "/ocean/projects/ccr200024p/zli27/sd_xray/sd/src/model_for_umap/xray_final_seed3_num1400"
    MODEL_PATH_FAIRGEN = "/ocean/projects/ccr200024p/zli27/sd_xray/sd/src/model_for_umap/xray_epoch5_seed3_num1400"

SAVE_NAME = f"umap_comparison_{MODALITY}_disease_only.png"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================
# 2. Disease Category Definition and Mapping Logic
# =========================================================================

def get_disease_mapping(modality):
    """
    Returns: (target class list, mapping function)
    Mapping function input: original folder name (e.g., 'African people psoriasis')
    Mapping function output: target class index (e.g., 3)
    """
    
    if modality == "skin":
        # Target classes (5 classes)
        target_classes = [
            "allergic contact dermatitis", "basal cell carcinoma", 
            "lichen planus", "psoriasis", "squamous cell carcinoma"
        ]
        def mapper(folder_name):
            name = folder_name.lower()
            for idx, disease in enumerate(target_classes):
                if disease in name: return idx
            return -1

    elif modality == "xray":
        # Target classes (5 classes, note the folder naming conventions)
        # Folders may be: female_COVID19, male_Lung_Opacity
        target_classes = [
            "covid19", "edema", "lung_opacity", "normal", "pleural_effusion"
        ]
        def mapper(folder_name):
            name = folder_name.lower()
            for idx, disease in enumerate(target_classes):
                # Simple substring matching, e.g., 'lung_opacity' in 'male_lung_opacity'
                if disease in name: return idx
            return -1

    elif modality == "mri":
        # Target classes (2 classes)
        # Folders: Demented_Age_Above_75, Nondemented_Age_Below_75
        target_classes = ["nondemented", "demented"]
        
        def mapper(folder_name):
            name = folder_name.lower()
            # Note the order! Match longer word nondemented first, otherwise demented will match both
            if "nondemented" in name: return 0
            if "demented" in name: return 1
            return -1
            
    else:
        raise ValueError("Unknown modality")
        
    return target_classes, mapper

# =========================================================================
# 3. Feature Extraction (including Label Remapping)
# =========================================================================

def get_features_and_remapped_labels(model_path, dataloader, mapper_func):
    print(f"\n--- Processing Model: {os.path.basename(model_path)} ---")
    
    # Load model
    try:
        model = ViTForImageClassification.from_pretrained(
            model_path, ignore_mismatched_sizes=True
        ).to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

    embeddings = []
    remapped_labels = []
    
    # Get original dataset class list (i.e., folder names)
    # dataloader.dataset is ImageFolder
    original_classes = dataloader.dataset.classes 
    
    print("Extracting features...")
    with torch.no_grad():
        for images, original_label_indices in tqdm(dataloader):
            images = images.to(device)
            
            # 1. Extract features
            outputs = model(images, output_hidden_states=True)
            cls_token = outputs.hidden_states[-1][:, 0, :]
            embeddings.append(cls_token.cpu().numpy())
            
            # 2. Remap labels in real-time (Original Index -> Folder Name -> Disease Index)
            for idx in original_label_indices:
                folder_name = original_classes[idx]
                new_idx = mapper_func(folder_name)
                if new_idx == -1:
                    print(f"Warning: Folder '{folder_name}' matched no disease!")
                remapped_labels.append(new_idx)
            
    return np.concatenate(embeddings), np.array(remapped_labels)

# =========================================================================
# 4. Main Program
# =========================================================================

def main():
    print(f"Generating Disease-Only UMAP for: {MODALITY}")
    
    # Get mapping configuration
    TARGET_CLASS_NAMES, label_mapper = get_disease_mapping(MODALITY)
    print(f"Target Disease Classes: {TARGET_CLASS_NAMES}")
    
    # Prepare data loader
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])
    
    if not os.path.exists(REAL_TEST_SET_ROOT):
        print("Error: Test set path not found.")
        return

    dataset = ImageFolder(REAL_TEST_SET_ROOT, transform=tfm)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # --- Extract features (with label merging) ---
    print("-> Vanilla Model")
    feats_vanilla, labels_vanilla = get_features_and_remapped_labels(MODEL_PATH_VANILLA, dataloader, label_mapper)
    
    print("-> FairGen Model")
    feats_fairgen, labels_fairgen = get_features_and_remapped_labels(MODEL_PATH_FAIRGEN, dataloader, label_mapper)
    
    # --- Compute UMAP ---
    print("\nRunning UMAP...")
    # To keep coordinate systems consistent, here we fit them separately (could also merge)
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='cosine', random_state=42)
    
    emb_vanilla = reducer.fit_transform(feats_vanilla)
    score_vanilla = silhouette_score(feats_vanilla, labels_vanilla, metric='cosine')
    
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='cosine', random_state=42)
    emb_fairgen = reducer.fit_transform(feats_fairgen)
    score_fairgen = silhouette_score(feats_fairgen, labels_fairgen, metric='cosine')
    
    # --- Plotting ---
    print("Plotting...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    colors = plt.get_cmap('tab10') # Only 5 or 2 colors now, won't have dozens
    
    def plot_on_ax(ax, embedding, labels, title, score):
        for i, name in enumerate(TARGET_CLASS_NAMES):
            idx = (labels == i)
            if np.sum(idx) > 0:
                ax.scatter(
                    embedding[idx, 0], embedding[idx, 1],
                    label=name.replace("_", " ").title(), 
                    color=colors(i),
                    alpha=0.6, s=20, edgecolor='none'
                )
        
        ax.set_title(f"{title}\nSilhouette Score: {score:.4f}", fontsize=16, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#cccccc')

        # --- New: Put legend inside the plot ---
        ax.legend(
            loc='best',           # Auto-find location with fewest points
            fontsize=16,          # Font size
            frameon=True,         # Show frame
            framealpha=0.8,       # Frame transparency (0.8) to prevent obscuring points
            edgecolor='gray'      # Frame color
        )

        # 1. Add axis labels
        ax.set_xlabel("", fontsize=12, fontweight='bold')
        ax.set_ylabel("", fontsize=12, fontweight='bold')

        # 3. Set tick label size (optional)
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        # 4. Darken frame color for clarity
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#333333") # Dark gray
            spine.set_linewidth(1.0)

    plot_on_ax(axes[0], emb_vanilla, labels_vanilla, "Baseline (Vanilla SD)", score_vanilla)
    plot_on_ax(axes[1], emb_fairgen, labels_fairgen, "FairGen (Ours)", score_fairgen)

    plt.tight_layout()

    
    plt.savefig(SAVE_NAME, dpi=300, bbox_inches='tight')
    print(f"Saved: {SAVE_NAME}")

if __name__ == "__main__":
    main()