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

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# -----------------------------
# Gender dictionary
# -----------------------------
gender_dict = {0: 'Above75', 1: 'Below75'}

proportion_list = [(0.5,0.5),(0.3,0.7),(0.4,0.6),(0.7,0.3),(0.6,0.4),(0.2,0.8),(0.8,0.2)]

# Custom Dataset with Dual Labels
from torch.utils.data import Dataset

class DualLabelDataset(Dataset):
    def __init__(self, root, transform=None):
        self.dataset = ImageFolder(root=root, transform=transform)
        self.transform = transform
        self.race_labels = {0: 'Above75', 1: 'Below75'}
        self.combined_labels = self._create_combined_labels()

    
    def _create_combined_labels(self):
        combined = []
        for _, class_idx in self.dataset.imgs:
            disease_label = class_idx // 2  
            race_label = class_idx % 2      
            
            combined_label = race_label * 10 + disease_label
            combined.append(combined_label)
        return combined
    

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, class_idx = self.dataset[idx]
        disease_label = class_idx // 2
        race_label = class_idx % 2
        
        return image, disease_label, race_label
    
    def get_by_race_and_disease(self):
        race_disease_dict = defaultdict(lambda: defaultdict(list))
        # print("Fetching race and disease data...")
        for idx in range(len(self.dataset)):
            image, class_idx = self.dataset[idx]
            
            disease_label = class_idx // 2
            race_label = class_idx % 2
            
            # print(f"Index: {idx}, Race Label: {race_label}, Disease Label: {disease_label}")
            race_disease_dict[race_label][disease_label].append(idx)
            
        return race_disease_dict

def sample_by_gender_proportion(gender_disease_dict, num, gender_proportions):
    """
    按性别比例采样，并在每个性别内均匀覆盖各疾病。
    """
    selected_indices = []
    total_proportion = sum(gender_proportions.values())
    gender_num_samples = {
        gender: int(num * gender_proportions[gender] / total_proportion)
        for gender in gender_proportions
    }

    for gender, num_samples in gender_num_samples.items():
        disease_dict = gender_disease_dict[gender]
        disease_count = max(len(disease_dict), 1)
        samples_per_disease = max(1, num_samples // disease_count)
        for _, indices in disease_dict.items():
            if len(indices) >= samples_per_disease:
                selected_indices.extend(random.sample(indices, samples_per_disease))
            else:
                selected_indices.extend(indices)
    return selected_indices

# -----------------------------
# Argparse
# -----------------------------
parser = argparse.ArgumentParser(description="XGBoost (reweight) with ResNet features")
parser.add_argument("--data", type=str, required=True, help="Path to the original data directory")
parser.add_argument("--aug_data", type=str, default=None, help="Path to the augmented data directory")
parser.add_argument("--output", type=str, required=True, help="(kept for compatibility; not used to save model here)")
parser.add_argument("--batchsize", type=int, default=64, help="Batch size for feature extraction")
parser.add_argument("--seed", type=int, default=3, help="Random seed")

# 兼容保留
parser.add_argument("--method", type=str, default="ff")
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--epochs", type=int, default=10)

# reweight / xgb 超参
parser.add_argument("--reweight_rounds", type=int, default=5, help="Number of reweight rounds")
parser.add_argument("--xgb_estimators", type=int, default=400)
parser.add_argument("--xgb_max_depth", type=int, default=6)
parser.add_argument("--xgb_lr", type=float, default=0.1)
parser.add_argument("--early_stopping_rounds", type=int, default=200)
args = parser.parse_args()

# -----------------------------
# Global setup
# -----------------------------
os.environ["WANDB_SILENT"] = "true"  
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
# Helper: feature extraction
# -----------------------------
@torch.no_grad()
def extract_features(loader, backbone):
    backbone.eval()
    feats, labels, genders = [], [], []
    for imgs, disease_labels, gender_labels in loader:
        imgs = imgs.to(device)
        out = backbone(imgs)                # [B, 2048]
        feats.append(out.cpu().numpy())
        labels.append(disease_labels.numpy())
        genders.append(gender_labels.numpy())
    X = np.concatenate(feats, axis=0)
    y = np.concatenate(labels, axis=0).astype(np.int32).ravel()
    g = np.concatenate(genders, axis=0).astype(np.int32).ravel()
    return X, y, g

# -----------------------------
# Main experiment loop
# -----------------------------
for seed in [args.seed]:
    for num in range(100, 900, 100):
        for proportions in proportion_list:
            gender_proportions = {0: proportions[0], 1: proportions[1]}

            wandb.init(
                project="xgb-mri-reweight",
                name=f"xgb_reweight_trial{seed}_scale{gender_proportions}_num{num}",
                config={**vars(args), "gender_proportions": gender_proportions, "num_aug": num}
            )

            # -----------------------------
            # Build datasets / loaders
            # -----------------------------
            train_dataset = DualLabelDataset(root=args.data, transform=transform)
            print(f"Total samples in raw dataset: {len(train_dataset)}")

            if args.aug_data:
                aug_dataset = DualLabelDataset(root=args.aug_data, transform=transform)
                print(f"Total samples in augmented dataset: {len(aug_dataset)}")
                gdict = aug_dataset.get_by_gender_and_disease()
                selected_idx = sample_by_gender_proportion(gdict, num * 4, gender_proportions)
                selected_subset = Subset(aug_dataset, selected_idx)
            else:
                selected_subset = None

            train_len = int(0.9 * len(train_dataset))
            val_len = len(train_dataset) - train_len
            train_subset, val_subset = random_split(
                train_dataset,
                [train_len, val_len],
                generator=torch.Generator().manual_seed(seed)
            )

            if selected_subset is not None:
                combined_train_subset = ConcatDataset([train_subset, selected_subset])
            else:
                combined_train_subset = train_subset

            train_loader = DataLoader(
                combined_train_subset, batch_size=args.batchsize,
                shuffle=False, num_workers=4, pin_memory=True
            )
            val_loader = DataLoader(
                val_subset, batch_size=args.batchsize,
                shuffle=False, num_workers=4, pin_memory=True
            )

            # -----------------------------
            # ResNet backbone → features
            # -----------------------------
            resnet = models.resnet50(pretrained=True)
            resnet.fc = torch.nn.Identity()  # 输出2048维
            resnet.to(device)

            print("Extracting features...")
            X_train, y_train, g_train = extract_features(train_loader, resnet)
            X_val,   y_val,   g_val   = extract_features(val_loader, resnet)

            # -----------------------------
            # XGBoost with dynamic reweight
            # -----------------------------
            n_classes = int(len(np.unique(y_train)))
            use_cuda = torch.cuda.is_available()


            group_weights = np.array([0.5, 0.5], dtype=np.float32)

            for rw_round in range(args.reweight_rounds):

                sample_weight = np.array([group_weights[gi] for gi in g_train], dtype=np.float32)

                if n_classes <= 2:
                    params = dict(
                        objective="binary:logistic",
                        eval_metric=["logloss", "error"],
                        n_estimators=args.xgb_estimators,
                        learning_rate=args.xgb_lr,
                        max_depth=args.xgb_max_depth,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        early_stopping_rounds=args.early_stopping_rounds,
                        tree_method="gpu_hist" if use_cuda else "hist",
                        random_state=seed
                    )
                else:
                    params = dict(
                        objective="multi:softprob",
                        num_class=n_classes,
                        eval_metric=["mlogloss", "merror"],
                        n_estimators=args.xgb_estimators,
                        learning_rate=args.xgb_lr,
                        max_depth=args.xgb_max_depth,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        early_stopping_rounds=args.early_stopping_rounds,
                        tree_method="gpu_hist" if use_cuda else "hist",
                        random_state=seed
                    )

                xgb = XGBClassifier(**params)
                xgb.fit(
                    X_train, y_train,
                    sample_weight=sample_weight,
                    eval_set=[(X_val, y_val)],  
                    verbose=False
                )

                if hasattr(xgb, "best_iteration") and xgb.best_iteration is not None:
                    it_range = (0, xgb.best_iteration + 1)
                    prob_tr = xgb.predict_proba(X_train, iteration_range=it_range)
                    prob_va = xgb.predict_proba(X_val,   iteration_range=it_range)
                else:
                    prob_tr = xgb.predict_proba(X_train)
                    prob_va = xgb.predict_proba(X_val)

                if n_classes <= 2:
                    y_train_pred = (prob_tr[:, 1] >= 0.5).astype(int)
                    y_val_pred   = (prob_va[:, 1] >= 0.5).astype(int)
                else:
                    y_train_pred = np.argmax(prob_tr, axis=1)
                    y_val_pred   = np.argmax(prob_va, axis=1)


                train_acc = accuracy_score(y_train, y_train_pred) * 100.0
                val_acc   = accuracy_score(y_val,   y_val_pred) * 100.0

                gender_correct = {0: 0, 1: 0}
                gender_total   = {0: 0, 1: 0}
                for yi, pi, gi in zip(y_val, y_val_pred, g_val):
                    gender_correct[gi] += int(yi == pi)
                    gender_total[gi]   += 1
                acc_list = [
                    100.0 * gender_correct[0] / max(gender_total[0], 1),
                    100.0 * gender_correct[1] / max(gender_total[1], 1),
                ]

                print(f"[RW {rw_round+1}/{args.reweight_rounds}] "
                      f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
                      f"{gender_dict[0]}: {acc_list[0]:.2f}% | {gender_dict[1]}: {acc_list[1]:.2f}%")

                # -----------------------------
                # wandb logging
                # -----------------------------
                wandb.log({
                    "rw_round": rw_round + 1,
                    "train_accuracy": train_acc,
                    "validation_accuracy": val_acc,
                    "Above75_accuracy": acc_list[0],
                    "Below75_accuracy": acc_list[1],
                    "n_estimators": xgb.get_params()["n_estimators"],
                    "max_depth": xgb.get_params()["max_depth"],
                    "learning_rate": xgb.get_params()["learning_rate"]
                })

                # 用本轮验证集分组 accuracy 的倒数（归一化）更新组权重
                eps = 1e-6
                inv = np.array([1.0 / max(a/100.0, eps) for a in acc_list], dtype=np.float32)
                group_weights = inv / inv.sum()

            wandb.finish()
            print("------------------------------")
