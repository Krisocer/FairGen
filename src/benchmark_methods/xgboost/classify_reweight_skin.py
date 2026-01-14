import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from collections import defaultdict
import random
import wandb
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# -----------------------------
# Dict / helpers
# -----------------------------
racc_dict = {0: 'African', 1: 'Asian', 2: 'Caucasian'}

def sample_by_race_proportion(race_disease_dict, num, race_proportions):
    selected_indices = []
    total_proportion = sum(race_proportions.values())
    race_num_samples = {race: int(num * race_proportions[race] / total_proportion)
                        for race in race_proportions}
    for race, num_samples in race_num_samples.items():
        disease_dict = race_disease_dict[race]
        disease_count = max(1, len(disease_dict))
        samples_per_disease = max(1, num_samples // disease_count)
        for _, indices in disease_dict.items():
            if len(indices) >= samples_per_disease:
                selected_indices.extend(random.sample(indices, samples_per_disease))
            else:
                selected_indices.extend(indices)
    return selected_indices

class DualLabelDataset(Dataset):
    def __init__(self, root, transform=None):
        self.dataset = ImageFolder(root=root, transform=transform)
        self.transform = transform
        self.combined_labels = self._create_combined_labels()  

    def _create_combined_labels(self):
        combined = []
        for _, class_idx in self.dataset.imgs:
            race_label = class_idx // 5
            disease_label = class_idx % 5
            combined.append(race_label * 10 + disease_label)
        return combined

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

def build_loader(ds, batch_size=64, shuffle=False, num_workers=8):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

@torch.no_grad()
def extract_features(backbone, loader, device):
    feats, ys, gs = [], [], []
    for x, y, g in loader:
        x = x.to(device)
        f = backbone(x)  # (B, 2048)
        feats.append(f.cpu().numpy())
        ys.append(y.numpy())
        gs.append(g.numpy())
    return np.concatenate(feats), np.concatenate(ys), np.concatenate(gs)

def group_accuracy(y_true, y_pred, groups, n_groups=3):
    corr = {i: 0 for i in range(n_groups)}
    tot = {i: 0 for i in range(n_groups)}
    for yt, yp, gg in zip(y_true, y_pred, groups):
        gi = int(gg)
        corr[gi] += int(yt == yp)
        tot[gi]  += 1
    accs = []
    for i in range(n_groups):
        accs.append(100.0 * corr[i] / max(1, tot[i]))
    return accs

# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser(description="ResNet50 features + XGBoost (multiclass) with reweight (minimal wandb logging)")
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--aug_data", type=str, default=None)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--batchsize", type=int, default=64)
parser.add_argument("--seed", type=int, default=3)

parser.add_argument("--reweight_rounds", type=int, default=5)
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

for seed in [args.seed]:
    for num in range(500, 1000, 100):

        wandb.init(
            project="xgb-skin-reweight",
            name=f"xgb_reweight_trial{seed}_num{num}",
            config={**vars(args)}
        )

        # -----------------------------
        # Datasets
        # -----------------------------
        base_ds = DualLabelDataset(root=args.data, transform=transform)

        if args.aug_data:
            aug_full = DualLabelDataset(root=args.aug_data, transform=transform)
            rd = aug_full.get_by_race_and_disease()
            race_props = {0: 1/3, 1: 1/3, 2: 1/3}
            sel_idx = sample_by_race_proportion(rd, num * 15, race_props)
            aug_ds = Subset(aug_full, sel_idx)
            train_source = ConcatDataset([base_ds, aug_ds])
        else:
            train_source = base_ds

        # -----------------------------
        # Stratified split on base set
        # -----------------------------
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
        train_val_idx, test_idx = next(splitter.split(np.zeros(len(base_ds)), base_ds.combined_labels))

        splitter_val = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
        tr_rel, va_rel = next(splitter_val.split(train_val_idx, np.array(base_ds.combined_labels)[train_val_idx]))
        train_idx = train_val_idx[tr_rel]
        val_idx   = train_val_idx[va_rel]

        train_subset = Subset(base_ds, train_idx)
        val_subset   = Subset(base_ds, val_idx)
        test_subset  = Subset(base_ds, test_idx)  

        if args.aug_data:
            combined_train = ConcatDataset([train_subset, aug_ds])
        else:
            combined_train = train_subset

        # -----------------------------
        # Frozen ResNet50 as feature extractor
        # -----------------------------
        resnet = models.resnet50(pretrained=True)
        resnet.fc = torch.nn.Identity()
        resnet.eval().to(device)

        train_loader = build_loader(combined_train, batch_size=args.batchsize, shuffle=False)
        val_loader   = build_loader(val_subset, batch_size=args.batchsize, shuffle=False)

        X_train, y_train, g_train = extract_features(resnet, train_loader, device)  # y: 0..4
        X_val,   y_val,   g_val   = extract_features(resnet, val_loader, device)

        # -----------------------------
        # XGBoost (multiclass) + reweight
        # -----------------------------

        group_weights = np.array([1/3, 1/3, 1/3], dtype=np.float32)

        for rw_round in range(args.reweight_rounds):
            sample_weight = np.array([group_weights[int(gg)] for gg in g_train], dtype=np.float32)

            xgb = XGBClassifier(
                n_estimators=args.xgb_estimators,
                max_depth=args.xgb_max_depth,
                learning_rate=args.xgb_lr,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                objective="multi:softprob",
                num_class=5,
                tree_method="gpu_hist" if torch.cuda.is_available() else "hist",
                random_state=seed,
                eval_metric="mlogloss"   
            )

            try:
                xgb.fit(
                    X_train, y_train,
                    sample_weight=sample_weight,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=args.early_stopping_rounds,  
                    verbose=False
                )
            except TypeError:

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

            y_tr_pred = np.argmax(prob_tr, axis=1)
            y_va_pred = np.argmax(prob_va, axis=1)

            train_acc = accuracy_score(y_train, y_tr_pred) * 100.0
            val_acc   = accuracy_score(y_val,   y_va_pred) * 100.0
            acc_list  = group_accuracy(y_val, y_va_pred, g_val, n_groups=3)  # [African, Asian, Caucasian]

            print(f"[RW {rw_round+1}/{args.reweight_rounds}] "
                  f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
                  f"African: {acc_list[0]:.2f}% | Asian: {acc_list[1]:.2f}% | Caucasian: {acc_list[2]:.2f}%")

            # -----------------------------
            # wandb
            # -----------------------------
            wandb.log({
                "train_accuracy": train_acc,
                "validation_accuracy": val_acc,
                "African_accuracy":  acc_list[0],
                "Asian_accuracy":    acc_list[1],
                "Caucasian_accuracy":acc_list[2],
                "n_estimators": xgb.get_params()["n_estimators"],
                "max_depth":   xgb.get_params()["max_depth"],
                "learning_rate": xgb.get_params()["learning_rate"]
            })

            eps = 1e-6
            inv = np.array([1.0 / max(a/100.0, eps) for a in acc_list], dtype=np.float32)
            group_weights = inv / inv.sum()

        wandb.finish()
