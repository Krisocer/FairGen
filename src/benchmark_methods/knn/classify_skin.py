import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from collections import defaultdict
import random
import wandb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Race dictionary
# -----------------------------
racc_dict = {0: 'African', 1: 'Asian', 2: 'Caucasian'}

# 比例组合
proportion_list = [
    (0.2, 0.2, 0.6), (0.2, 0.3, 0.5), (0.2, 0.4, 0.4),
    (0.2, 0.5, 0.3), (0.2, 0.6, 0.2), (0.333, 0.333, 0.333),
    (0.3, 0.2, 0.5), (0.3, 0.3, 0.4), (0.3, 0.4, 0.3), (0.3, 0.5, 0.2),
    (0.4, 0.2, 0.4), (0.4, 0.3, 0.3),
    (0.4, 0.4, 0.2), (0.5, 0.2, 0.3),
    (0.5, 0.3, 0.2), (0.6, 0.2, 0.2)
]


# -----------------------------
# Dataset with dual labels
# -----------------------------
class DualLabelDataset(Dataset):
    def __init__(self, root, transform=None):
        self.dataset = ImageFolder(root=root, transform=transform)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, class_idx = self.dataset[idx]
        race_label = class_idx // 5 
        disease_label = class_idx % 5
        return image, disease_label, race_label

    def get_by_race_and_disease(self):
        race_disease_dict = defaultdict(lambda: defaultdict(list))
        for idx in range(len(self.dataset)):
            _, class_idx = self.dataset[idx]
            race_label = class_idx // 5
            disease_label = class_idx % 5
            race_disease_dict[race_label][disease_label].append(idx)
        return race_disease_dict


# -----------------------------
# 采样函数
# -----------------------------
def sample_by_race_proportion(race_disease_dict, num, race_proportions):
    selected_indices = []
    total_proportion = sum(race_proportions.values())
    race_num_samples = {race: int(num * race_proportions[race] / total_proportion)
                        for race in race_proportions}
    for race, num_samples in race_num_samples.items():
        disease_dict = race_disease_dict[race]
        disease_count = len(disease_dict)
        samples_per_disease = num_samples // disease_count
        for disease, indices in disease_dict.items():
            if len(indices) >= samples_per_disease:
                selected_indices.extend(random.sample(indices, samples_per_disease))
            else:
                selected_indices.extend(indices)
    return selected_indices


# -----------------------------
# Argument Parser
# -----------------------------
parser = argparse.ArgumentParser(description="KNN Skin Fairness Classification")
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--aug_data", type=str, default=None)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--batchsize", type=int, default=64)
parser.add_argument("--seed", type=int, default=3)
args = parser.parse_args()

# -----------------------------
# Global setup
# -----------------------------
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# Main experiment loop
# -----------------------------
for seed in [3]:
    args.seed = seed
    for num in range(500, 1000, 100):
        for idx, proportions in enumerate(proportion_list):
            race_proportions = {0: proportions[0], 1: proportions[1], 2: proportions[2]}

            wandb.init(
                project="knn-skin",
                name=f"knn_trial{args.seed}_scale{race_proportions}_num{num}",
                config={**vars(args)}
            )

            # -----------------------------
            # Dataset setup
            # -----------------------------
            train_dataset = DualLabelDataset(root=args.data, transform=transform)
            print(f"Total samples in raw dataset: {len(train_dataset)}")

            if args.aug_data:
                aug_dataset = DualLabelDataset(root=args.aug_data, transform=transform)
                print(f"Total samples in augmented dataset: {len(aug_dataset)}")
                race_disease_dict = aug_dataset.get_by_race_and_disease()
                selected_indices = sample_by_race_proportion(race_disease_dict, num * 15, race_proportions)
                selected_subset = Subset(aug_dataset, selected_indices)
            else:
                selected_subset = None

            train_len = int(0.9 * len(train_dataset))
            val_len = len(train_dataset) - train_len
            local_seed = (args.seed * 1000003 + num * 97 + int(proportions[0]*1000)) % (2**32)
            train_subset, val_subset = random_split(train_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(local_seed))

            if args.aug_data:
                combined_train_subset = ConcatDataset([train_subset, selected_subset])
            else:
                combined_train_subset = train_subset

            train_loader = DataLoader(combined_train_subset, batch_size=args.batchsize, shuffle=False, num_workers=4)
            val_loader = DataLoader(val_subset, batch_size=args.batchsize, shuffle=False, num_workers=4)

            # -----------------------------
            # Feature extraction backbone
            # -----------------------------
            resnet = models.resnet50(pretrained=True)
            resnet.fc = torch.nn.Identity()  # 去掉分类头
            resnet.eval().to(device)

            def extract_features(loader, model):
                features, labels, races = [], [], []
                with torch.no_grad():
                    for imgs, disease_labels, race_labels in loader:
                        imgs = imgs.to(device)
                        out = model(imgs)
                        features.append(out.cpu().numpy())
                        labels.append(disease_labels.numpy())
                        races.append(race_labels.numpy())
                return np.concatenate(features), np.concatenate(labels), np.concatenate(races)

            print("Extracting features...")
            X_train, y_train, r_train = extract_features(train_loader, resnet)
            X_val, y_val, r_val = extract_features(val_loader, resnet)

            # -----------------------------
            # Train KNN classifier
            # -----------------------------
            print("Training KNN classifier...")
            knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='cosine', n_jobs=-1)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_val)

            # -----------------------------
            # Evaluate
            # -----------------------------
            overall_acc = accuracy_score(y_val, y_pred) * 100
            print(f"\nOverall Accuracy: {overall_acc:.2f}%")

            race_correct = {0: 0, 1: 0, 2: 0}
            race_total = {0: 0, 1: 0, 2: 0}
            for yi, pi, ri in zip(y_val, y_pred, r_val):
                race_correct[ri] += int(yi == pi)
                race_total[ri] += 1

            acc_list = []
            for race in range(3):
                acc = 100 * race_correct[race] / max(race_total[race], 1)
                print(f"Accuracy for {racc_dict[race]}: {acc:.2f}%")
                acc_list.append(acc)

            # -----------------------------
            # Log to wandb
            # -----------------------------
            wandb.log({
                "validation_accuracy": overall_acc,
                "African_accuracy": acc_list[0],
                "Asian_accuracy": acc_list[1],
                "Caucasian_accuracy": acc_list[2],
            })

            wandb.finish()
            print("------------------------------")
