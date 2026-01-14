import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from collections import defaultdict
import random
import wandb
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# -----------------------------
# Config
# -----------------------------
os.environ["WANDB_SILENT"] = "true"
gender_dict = {0: 'Female', 1: 'Male'}

proportion_list = [(0.5,0.5),(0.3,0.7),(0.4,0.6),(0.7,0.3),(0.6,0.4),(0.2,0.8),(0.8,0.2)]

# -----------------------------
# Dataset (gender * 5 + disease)
# -----------------------------
class DualLabelDataset(Dataset):
    """
    ImageFolder 假设类索引为: gender(0/1)*5 + disease(0..4)
    训练/评估只用 disease 标签(5 类)；gender 仅用于采样与公平性评估。
    """
    def __init__(self, root, transform=None):
        self.dataset = ImageFolder(root=root, transform=transform)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, class_idx = self.dataset[idx]
        gender_label = class_idx // 5
        disease_label = class_idx % 5
        return image, disease_label, gender_label

    def get_by_gender_and_disease(self):
        d = defaultdict(lambda: defaultdict(list))  # {gender: {disease: [idx]}}
        # 用 samples 避免重复解码
        for idx, (_, class_idx) in enumerate(self.dataset.samples):
            gender_label = class_idx // 5
            disease_label = class_idx % 5
            d[gender_label][disease_label].append(idx)
        return d

# -----------------------------
# Helpers
# -----------------------------
def sample_by_gender_proportion(gender_disease_dict, num, gender_proportions):
    """
    按性别比例采样，并在每个性别内尽量均匀覆盖 5 个疾病。
    """
    selected_indices = []
    total_prop = sum(gender_proportions.values())
    gender_num_samples = {g: int(num * gender_proportions[g] / max(1e-12, total_prop))
                          for g in gender_proportions}
    for g, n in gender_num_samples.items():
        disease_dict = gender_disease_dict[g]
        k = max(1, len(disease_dict))
        per = max(1, n // k)
        for _, idxs in disease_dict.items():
            if len(idxs) >= per:
                selected_indices.extend(random.sample(idxs, per))
            else:
                selected_indices.extend(idxs)
    return selected_indices

def build_loader(ds, batch_size=64, shuffle=False, num_workers=4):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)

@torch.no_grad()
def extract_features(backbone, loader, device):
    feats, ys, gs = [], [], []
    backbone.eval()
    for x, y, g in loader:
        x = x.to(device)
        f = backbone(x)      # (B, 2048)
        feats.append(f.cpu().numpy())
        ys.append(y.numpy())
        gs.append(g.numpy())
    return np.concatenate(feats), np.concatenate(ys), np.concatenate(gs)

def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); random.seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser(description="ResNet50 features + XGBoost (5-class, no reweight) with gender fairness logging")
parser.add_argument("--data", type=str, required=True, help="Base data path")
parser.add_argument("--aug_data", type=str, default=None, help="Augmented data path")
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--batchsize", type=int, default=64)
parser.add_argument("--seed", type=int, default=6)

# XGBoost
parser.add_argument("--xgb_estimators", type=int, default=400)
parser.add_argument("--xgb_max_depth", type=int, default=6)
parser.add_argument("--xgb_lr", type=float, default=0.1)
parser.add_argument("--early_stopping_rounds", type=int, default=200)
args = parser.parse_args()

# -----------------------------
# Setup
# -----------------------------
set_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# Sweep over seed/num/proportions
# -----------------------------
for seed in [3]:
    set_seed(seed)

    for num in range(1400, 3000, 200):
        for proportions in proportion_list:
            gender_proportions = {0: proportions[0], 1: proportions[1]}

            wandb.init(
                project="xgb-xray",
                name=f"xgb_trial{seed}_scale{gender_proportions}_num{num}",
                config={**vars(args), "gender_scale": gender_proportions, "num": num, "seed": seed}
            )

            # -------- dataset --------
            base_ds = DualLabelDataset(root=args.data, transform=transform)

            if args.aug_data:
                aug_full = DualLabelDataset(root=args.aug_data, transform=transform)
                print(f"Total samples in augmented dataset: {len(aug_full)}")
                gdict = aug_full.get_by_gender_and_disease()
                sel_idx = sample_by_gender_proportion(gdict, num * 10, gender_proportions)
                aug_ds = Subset(aug_full, sel_idx)
                train_source = ConcatDataset([base_ds, aug_ds])
            else:
                train_source = base_ds

            # -------- random_split --------
            train_len = int(0.9 * len(base_ds))
            val_len = len(base_ds) - train_len
            train_subset, val_subset = random_split(
                base_ds, [train_len, val_len],
                generator=torch.Generator().manual_seed(seed)
            )

            if args.aug_data:
                combined_train = ConcatDataset([train_subset, aug_ds])
            else:
                combined_train = train_subset

            # -------- ResNet50 feature extractor --------
            resnet = models.resnet50(pretrained=True)
            resnet.fc = torch.nn.Identity()
            resnet.to(device).eval()

            train_loader = build_loader(combined_train, batch_size=args.batchsize, shuffle=False)
            val_loader   = build_loader(val_subset,   batch_size=args.batchsize, shuffle=False)

            X_train, y_train, g_train = extract_features(resnet, train_loader, device)  # y: 0..4
            X_val,   y_val,   g_val   = extract_features(resnet, val_loader,   device)

            # -------- XGBoost（multi:softprob, no reweight）--------
            use_cuda = torch.cuda.is_available()
            xgb = XGBClassifier(
                n_estimators=args.xgb_estimators,
                max_depth=args.xgb_max_depth,
                learning_rate=args.xgb_lr,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                objective="multi:softprob",
                num_class=5,
                eval_metric=['mlogloss', 'merror'],               
                early_stopping_rounds=args.early_stopping_rounds, 
                tree_method="gpu_hist" if use_cuda else "hist",
                random_state=seed
            )

            xgb.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False
            )

            best_iter = xgb.best_iteration if getattr(xgb, "best_iteration", None) is not None else None
            if best_iter is not None:
                proba_tr = xgb.predict_proba(X_train, iteration_range=(0, best_iter + 1))
                proba_va = xgb.predict_proba(X_val,   iteration_range=(0, best_iter + 1))
            else:
                proba_tr = xgb.predict_proba(X_train)
                proba_va = xgb.predict_proba(X_val)

            y_tr_pred = np.argmax(proba_tr, axis=1)
            y_va_pred = np.argmax(proba_va, axis=1)

            # ---- （Female/Male）----
            train_acc = accuracy_score(y_train, y_tr_pred) * 100.0
            val_acc   = accuracy_score(y_val,   y_va_pred) * 100.0

            gender_correct = {0: 0, 1: 0}
            gender_total   = {0: 0, 1: 0}
            for yt, yp, gg in zip(y_val, y_va_pred, g_val):
                gi = int(gg)
                gender_correct[gi] += int(yt == yp)
                gender_total[gi]   += 1
            female_acc = 100.0 * gender_correct[0] / max(1, gender_total[0])
            male_acc   = 100.0 * gender_correct[1] / max(1, gender_total[1])

            print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
                  f"Female: {female_acc:.2f}% | Male: {male_acc:.2f}%")

            # ---- wandb ----
            wandb.log({
                "train_accuracy": train_acc,
                "validation_accuracy": val_acc,
                "female_accuracy": female_acc,
                "male_accuracy": male_acc,
                "n_estimators": xgb.get_params()["n_estimators"],
                "max_depth": xgb.get_params()["max_depth"],
                "learning_rate": xgb.get_params()["learning_rate"]
            })

            # os.makedirs(args.output, exist_ok=True)
            # xgb.get_booster().save_model(os.path.join(args.output, f"xgb_xray_best_scale{gender_proportions}_num{num}.json"))

            wandb.finish()
            print("------------------------------")
