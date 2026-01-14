import argparse
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import numpy as np
# from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
import matplotlib.pyplot as plt
import random
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
# import seaborn as sns
from collections import defaultdict

# Define race dictionary
racc_dict = {0: 'Female', 1: 'Male'}

# Custom Dataset with Dual Labels
from torch.utils.data import Dataset


def sample_by_race_proportion(race_disease_dict, num, race_proportions):
        selected_indices = []
   
        
        total_proportion = sum(race_proportions.values())
        race_num_samples = {race: int(num * race_proportions[race] / total_proportion) for race in race_proportions}
    
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


class DualLabelDataset(Dataset):
    def __init__(self, root, transform=None):
        self.dataset = ImageFolder(root=root, transform=transform)
        self.transform = transform
        self.race_labels = {0: 'Female', 1: 'Male'}
        self.combined_labels = self._create_combined_labels()

    def _create_combined_labels(self):
        combined = []
        for _, class_idx in self.dataset.imgs:
            race_label = class_idx // 5  # Adjust based on your class indexing
            disease_label = class_idx % 5
            combined_label = race_label * 10 + disease_label  # Unique combination
            combined.append(combined_label)
        return combined

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, class_idx = self.dataset[idx]
        race_label = class_idx // 5
        disease_label = class_idx % 5
        return image, disease_label, race_label
    
    def get_by_race_and_disease(self):
        race_disease_dict = defaultdict(lambda: defaultdict(list))  # {race: {disease: [samples]}}
        
        for idx in range(len(self.dataset)):
            image, class_idx = self.dataset[idx]
            race_label = class_idx // 5
            disease_label = class_idx % 5
            race_disease_dict[race_label][disease_label].append(idx)
            
        return race_disease_dict
    
    


# Initialize the parser
parser = argparse.ArgumentParser(description="ViT Disease Classification with Fairness")
parser.add_argument("--data", type=str, required=True, help="Path to the original data directory")
parser.add_argument("--aug_data", type=str, default=None, help="Path to the augmented data directory")
parser.add_argument("--output", type=str, required=True, help="Path to the output directory for the model")
parser.add_argument("--method", type=str,default="ff", help="Method to use for training ('ff' for full fine-tuning, 'lp' for linear probing)")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
parser.add_argument("--batchsize", type=int, default=64, help="Batch size for training")
parser.add_argument("--seed", type=int, default=0, help="Random seed")

# Parse the arguments
args = parser.parse_args()
for seed in [3]:
    args.seed = seed
    for num in range(1400,3000,200):

        # Initialize wandb
        wandb.init(
            project="resnet-xray",
            name=f"{args.method}_lr{args.lr}_bs{args.batchsize}_aug{os.path.basename(args.aug_data) if args.aug_data else 'no_aug'}_trial{args.seed}_num{num}_reweight",
            config={**vars(args)}
        )

        # Set random seed for reproducibility
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        from torchvision import models

        # 加载预训练的 ResNet50
        model = models.resnet50(pretrained=True)

        # 替换最后的全连接层 (fc)，根据你的类别数设置输出（此处是2类）
        num_features = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(num_features, 5)
        )

        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
        ])


        # Initialize datasets
        train_dataset = DualLabelDataset(root=args.data, transform=transform)
        print(f"Total samples in raw dataset: {len(train_dataset)}")

        # Handle augmented dataset if provided
        if args.aug_data:
            aug_train_dataset = DualLabelDataset(root=args.aug_data, transform=transform)
            print(f"Total samples in agumentation dataset: {len(aug_train_dataset)}")
            race_disease_dict = aug_train_dataset.get_by_race_and_disease()
            race_proportions = {0: 0.5, 1: 0.5}
            selected_indices = sample_by_race_proportion(race_disease_dict, num * 10, race_proportions)
            aug_train_dataset = Subset(aug_train_dataset, selected_indices)

        # Stratified splitting
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=args.seed)
        train_val_idx, test_idx = next(splitter.split(np.zeros(len(train_dataset)), train_dataset.combined_labels))

        splitter_val = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=args.seed)  
        train_idx_split, val_idx_split = next(splitter_val.split(train_val_idx, np.array(train_dataset.combined_labels)[train_val_idx]))

        # Create subsets
        train_subset = Subset(train_dataset, train_val_idx[train_idx_split])
        val_subset = Subset(train_dataset, train_val_idx[val_idx_split])
        test_subset = Subset(train_dataset, test_idx)

        # Handle augmented training data
        if args.aug_data:
            # aug_train_subset = Subset(aug_train_dataset, train_val_idx[train_idx_split])  # Adjust based on augmentation strategy
            combined_train_subset = ConcatDataset([train_subset, aug_train_dataset])
        else:
            combined_train_subset = train_subset


        # DataLoaders
        train_loader = DataLoader(
            combined_train_subset,
            batch_size=args.batchsize,
            num_workers=16
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=args.batchsize,
            shuffle=False,
            num_workers=16
        )

        test_loader = DataLoader(
            test_subset,
            batch_size=args.batchsize,
            shuffle=False,
            num_workers=16
        )

        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Define optimizer and loss with class weights
        if args.method == "ff":
            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        elif args.method == "lp":
            optimizer = AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=1e-5)
        else:
            raise ValueError("Invalid method")

        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)


        criterion = CrossEntropyLoss()

        # Training Loop with Dynamic Sampling Based on Validation Accuracies
        best_val_accuracy = 0.0
        patience = 3
        trigger_times = 0

        for epoch in range(args.epochs):
            # If it's the first epoch, use uniform class weights
            if epoch == 0:
                class_weights = torch.ones(2)  # Assuming 2 races
            else:
                # Use per-race validation accuracies to compute class weights
                # Adding a small epsilon to avoid division by zero
                class_weights = 1 / (torch.tensor(acc_list) + 1e-6)
                # Normalize class weights
                class_weights = class_weights / class_weights.sum()

            # Get race labels for all samples in the combined training subset
            race_labels = []
            if isinstance(combined_train_subset, ConcatDataset):
                for ds in combined_train_subset.datasets:
                    for idx in range(len(ds)):
                        _, _, race_label = ds[idx]
                        race_labels.append(race_label)
            else:
                for idx in range(len(combined_train_subset)):
                    _, _, race_label = combined_train_subset[idx]
                    race_labels.append(race_label)

            # Assign sample weights based on race labels
            sample_weights = [class_weights[race_label] for race_label in race_labels]

            # Create a WeightedRandomSampler
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

            # Update the DataLoader with the new sampler
            train_loader = DataLoader(
                combined_train_subset,
                batch_size=args.batchsize,
                sampler=sampler,
                num_workers=16
            )

            # Training
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                inputs, disease_labels, _ = batch  # Ignore race labels during training
                inputs = inputs.to(device)
                disease_labels = disease_labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, disease_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            scheduler.step()
            print(f'Epoch {epoch+1}/{args.epochs} - Loss: {epoch_loss:.4f}')

            # Training Accuracy
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in train_loader:
                    inputs, disease_labels, _ = batch
                    inputs = inputs.to(device)
                    disease_labels = disease_labels.to(device)

                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += disease_labels.size(0)
                    correct += (predicted == disease_labels).sum().item()

            train_accuracy = 100 * correct / total
            print(f'Train Accuracy: {train_accuracy:.2f}%')

            # Validation Accuracy and Fairness Metrics
            race_correct = {0: 0, 1: 0}
            race_total = {0: 0, 1: 0}
            y_true = []
            y_pred = []
            race_labels_batch_all = []

            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, disease_labels, race_labels_batch) in enumerate(val_loader):
                    inputs = inputs.to(device)
                    disease_labels = disease_labels.to(device)

                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)

                    total += disease_labels.size(0)
                    correct += (predicted == disease_labels).sum().item()

                    # Collect labels for fairness metrics
                    y_true.extend(disease_labels.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
                    race_labels_batch_all.extend(race_labels_batch.cpu().numpy())

                    # Per-race accuracy
                    for i in range(len(disease_labels)):
                        race = race_labels_batch[i].item()
                        race_correct[race] += (predicted[i] == disease_labels[i]).item()
                        race_total[race] += 1

            overall_accuracy = 100 * correct / total
            print(f'Validation Accuracy: {overall_accuracy:.2f}%')

            # Calculate and print accuracy for each race group
            acc_list = []
            for race in range(2):
                if race_total[race] > 0:
                    race_accuracy = 100 * race_correct[race] / race_total[race]
                    print(f'Accuracy for {racc_dict[race]} people: {race_accuracy:.2f}%')
                    acc_list.append(race_accuracy)
                else:
                    print(f'No samples for Race {race}')
                    acc_list.append(0.0)




            # Log metrics to wandb
            wandb.log({
                "epoch": epoch+1,
                "loss": epoch_loss,
                "train_accuracy": train_accuracy,
                "validation_accuracy": overall_accuracy,
                "female_accuracy": acc_list[0],
                "male_accuracy": acc_list[1],
            })

            # Save the best model
            if overall_accuracy > best_val_accuracy:
                best_val_accuracy = overall_accuracy
                # model.save_pretrained(os.path.join("model", args.output, "best_model"))
                print("Best model saved!")

            # Early Stopping (Optional)
            '''
            if overall_accuracy > best_val_accuracy:
                best_val_accuracy = overall_accuracy
                trigger_times = 0
                model.save_pretrained(os.path.join("model", args.output, "best_model"))
                print("Best model saved!")
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print("Early stopping triggered!")
                    break
            '''

        '''
        # Final Model Save
        # model.save_pretrained(os.path.join("model", args.output, "final_model"))

        # Test Evaluation Phase
        print("\n--- Test Evaluation ---")



        model.eval()

        # Initialize metrics for test set
        race_correct_test = {0: 0, 1: 0, 2: 0}
        race_total_test = {0: 0, 1: 0, 2: 0}
        y_true_test = []
        y_pred_test = []
        race_labels_test = []

        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for batch_idx, (inputs, disease_labels, race_labels_batch) in enumerate(test_loader):
                inputs = inputs.to(device)
                disease_labels = disease_labels.to(device)

                outputs = model(pixel_values=inputs)
                _, predicted = torch.max(outputs.logits, 1)

                total_test += disease_labels.size(0)
                correct_test += (predicted == disease_labels).sum().item()

                # Collect labels for fairness metrics
                y_true_test.extend(disease_labels.cpu().numpy())
                y_pred_test.extend(predicted.cpu().numpy())
                race_labels_test.extend(race_labels_batch.cpu().numpy())

                # Per-race accuracy
                for i in range(len(disease_labels)):
                    race = race_labels_batch[i].item()
                    race_correct_test[race] += (predicted[i] == disease_labels[i]).item()
                    race_total_test[race] += 1

        # Calculate overall test accuracy
        overall_test_accuracy = 100 * correct_test / total_test
        print(f'\nTest Accuracy: {overall_test_accuracy:.2f}%')

        # Calculate and print accuracy for each race group
        acc_test_list = []
        for race in range(2):
            if race_total_test[race] > 0:
                race_accuracy = 100 * race_correct_test[race] / race_total_test[race]
                print(f'Accuracy for {racc_dict[race]} people: {race_accuracy:.2f}%')
                acc_test_list.append(race_accuracy)
            else:
                print(f'No samples for Race {race}')
                acc_test_list.append(0.0)



        # Log test metrics to wandb
        wandb.log({
            "Test_accuracy": overall_test_accuracy,
            "Test_African_accuracy": acc_test_list[0],
            "Test_Asian_accuracy": acc_test_list[1],
            "Test_Caucasian_accuracy": acc_test_list[2]
        })
        '''

        wandb.finish()