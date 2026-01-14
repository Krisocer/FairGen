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
# Config
# -----------------------------
os.environ["WANDB_SILENT"] = "true"
racc_dict = {0: 'African', 1: 'Asian', 2: 'Caucasian'}

proportion_list = [
    (0.2, 0.2, 0.6), (0.2, 0.3, 0.5), (0.2, 0.4, 0.4),
    (0.2, 0.5, 0.3), (0.2, 0.6, 0.2), (0.333, 0.333, 0.333),
    (0.3, 0.2, 0.5), (0.3, 0.3, 0.4), (0.3, 0.4, 0.3), (0.3, 0.5, 0.2),
    (0.4, 0.2, 0.4), (0.4, 0.3, 0.3), (0.4, 0.4, 0.2),
    (0.5, 0.2, 0.3), (0.5, 0.3, 0.2), (0.6, 0.2, 0.2)
]

# -----------------------------
# Dataset with dual labels
# -----------------------------
class DualLabelDataset(Dataset):

    def __init__(self, root, transform=None):
        self.dataset = ImageFolder(root=root, transform=transform)
        self.transform = transform
        self.combined_labels = self._create_combined_labels()  # race*10 + disease

    def _create_combined_labels(self):
        combined = []
        for _, class_idx in self.dataset.samples:  
            race_label = class_idx // 5
            disease_label = class_idx % 5
            combined.append(race_label * 15 + disease_label)
        return combined

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, class_idx = self.dataset[idx]
        race_label = class_idx // 5
        disease_label = class_idx % 5
        return image, disease_label, race_label

    def get_by_race_and_disease(self):
        d = defaultdict(lambda: defaultdict(list))
        for idx, (_, class_idx) in enumerate(self.dataset.samples):
            race_label = class_idx // 5
            disease_label = class_idx % 5
            d[race_label][disease_label].append(idx)
        return d

# -----------------------------
# Helpers
# -----------------------------
def sample_by_race_proportion(race_disease_dict, num, race_proportions):
    """按人群比例采样，并在每个 race 内尽量均衡覆盖 5 个疾病。"""
    selected_indices = []
    total_prop = sum(race_proportions.values())
    race_num_samples = {r: int(num * race_proportions[r] / max(1e-12, total_prop))
                        for r in race_proportions}
    for race, n in race_num_samples.items():
        disease_dict = race_disease_dict[race]
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
    for x, y, g in loader:
        x = x.to(device)
        f = backbone(x)        # (B, 2048)
        feats.append(f.cpu().numpy())
        ys.append(y.numpy())
        gs.append(g.numpy())
    return np.concatenate(feats), np.concatenate(ys), np.concatenate(gs)

def group_accuracy(y_true, y_pred, groups, n_groups=3):
    corr = {i: 0 for i in range(n_groups)}
    tot  = {i: 0 for i in range(n_groups)}
    for yt, yp, gg in zip(y_true, y_pred, groups):
        gi = int(gg)
        corr[gi] += int(yt == yp)
        tot[gi]  += 1
    return [100.0 * corr[i] / max(1, tot[i]) for i in range(n_groups)]

# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser(description="ResNet50 features + XGBoost (multi-class, no reweight) with fairness logging")
parser.add_argument("--data", type=str, required=True, help="Path to base data")
parser.add_argument("--aug_data", type=str, default=None, help="Path to augmented data")
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--batchsize", type=int, default=64)
parser.add_argument("--seed", type=int, default=6)

# XGB 
parser.add_argument("--xgb_estimators", type=int, default=400)
parser.add_argument("--xgb_max_depth", type=int, default=6)
parser.add_argument("--xgb_lr", type=float, default=0.1)
parser.add_argument("--early_stopping_rounds", type=int, default=200)
args = parser.parse_args()

# -----------------------------
# Setup
# -----------------------------
def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); random.seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# Main sweep
# -----------------------------
for seed in [3]:
    set_seed(seed)

    for num in range(500, 1000, 100):
        for proportions in proportion_list:
            race_proportions = {0: proportions[0], 1: proportions[1], 2: proportions[2]}

            wandb.init(
                project="xgb-skin",
                name=f"xgb_trial{seed}_scale{race_proportions}_num{num}",
                config={**vars(args), "race_scale": race_proportions, "num": num, "seed": seed}
            )

            # ---------- datasets ----------
            base_ds = DualLabelDataset(root=args.data, transform=transform)

            if args.aug_data:
                aug_full = DualLabelDataset(root=args.aug_data, transform=transform)
                race_dict = aug_full.get_by_race_and_disease()
                sel_idx = sample_by_race_proportion(race_dict, num * 15, race_proportions)
                aug_ds = Subset(aug_full, sel_idx)
            else:
                aug_ds = None

            # ---------- stratified split on base set ----------
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
            train_val_idx, test_idx = next(splitter.split(np.zeros(len(base_ds)), base_ds.combined_labels))
            splitter_val = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
            tr_rel, va_rel = next(splitter_val.split(train_val_idx, np.array(base_ds.combined_labels)[train_val_idx]))
            train_idx = train_val_idx[tr_rel]; val_idx = train_val_idx[va_rel]

            train_subset = Subset(base_ds, train_idx)
            val_subset   = Subset(base_ds, val_idx)

            if aug_ds is not None:
                combined_train = ConcatDataset([train_subset, aug_ds])
            else:
                combined_train = train_subset

            # ---------- frozen ResNet50 -> features ----------
            resnet = models.resnet50(pretrained=True)
            resnet.fc = torch.nn.Identity()
            resnet.eval().to(device)

            train_loader = build_loader(combined_train, batch_size=args.batchsize, shuffle=False)
            val_loader   = build_loader(val_subset, batch_size=args.batchsize, shuffle=False)

            X_train, y_train, g_train = extract_features(resnet, train_loader, device)  # y: 0..4
            X_val,   y_val,   g_val   = extract_features(resnet, val_loader, device)

            # ---------- XGBoost (multi:softprob) ----------
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

            results = xgb.evals_result() if hasattr(xgb, "evals_result") else xgb.get_booster().evals_result()
            tr_ll = results['validation_0']['mlogloss']; tr_er = results['validation_0']['merror']
            va_ll = results['validation_1']['mlogloss']; va_er = results['validation_1']['merror']
            for r, (a, b) in enumerate(zip(tr_ll, va_ll), start=1):
                wandb.log({"round": r, "train_mlogloss": a, "val_mlogloss": b})
            for r, (a, b) in enumerate(zip(tr_er, va_er), start=1):
                wandb.log({"round": r, "train_merror": a, "val_merror": b})

            best_iter = xgb.best_iteration if getattr(xgb, "best_iteration", None) is not None else (len(va_ll) - 1)
            proba_tr = xgb.predict_proba(X_train, iteration_range=(0, best_iter + 1))
            proba_va = xgb.predict_proba(X_val,   iteration_range=(0, best_iter + 1))
            y_tr_pred = np.argmax(proba_tr, axis=1)
            y_va_pred = np.argmax(proba_va, axis=1)

            train_acc = accuracy_score(y_train, y_tr_pred) * 100.0
            val_acc   = accuracy_score(y_val,   y_va_pred) * 100.0
            grp_accs  = group_accuracy(y_val, y_va_pred, g_val, n_groups=3)  # [African, Asian, Caucasian]

            # ----wandb----
            wandb.log({
                "train_accuracy": train_acc,
                "validation_accuracy": val_acc,
                "African_accuracy":  grp_accs[0],
                "Asian_accuracy":    grp_accs[1],
                "Caucasian_accuracy":grp_accs[2],
                "n_estimators": xgb.get_params()["n_estimators"],
                "max_depth":   xgb.get_params()["max_depth"],
                "learning_rate": xgb.get_params()["learning_rate"],
                "best_iteration": best_iter
            })

            # os.makedirs(args.output, exist_ok=True)
            # xgb.get_booster().save_model(os.path.join(args.output, f"xgb_best_scale{proportions}_num{num}.json"))

            wandb.finish()
            print("------------------------------")
