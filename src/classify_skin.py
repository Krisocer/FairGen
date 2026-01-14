import argparse
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import numpy as np
# from fairlearn.metrics import MetricFrame,  demographic_parity_difference, equalized_odds_difference
import matplotlib.pyplot as plt
import random
from collections import Counter
from collections import defaultdict
import time
import random


# Define race dictionary
racc_dict = {0: 'African', 1: 'Asian', 2: 'Caucasian'}

proportion_list = [
    
    (0.2, 0.2, 0.6), (0.2, 0.3, 0.5), (0.2, 0.4, 0.4),
    (0.2, 0.5, 0.3), (0.2, 0.6, 0.2), (0.333, 0.333, 0.333),
    (0.3, 0.2, 0.5), (0.3, 0.3, 0.4), (0.3, 0.4, 0.3), (0.3, 0.5, 0.2),
    (0.4, 0.2, 0.4), (0.4, 0.3, 0.3),
    (0.4, 0.4, 0.2),  (0.5, 0.2, 0.3),
    (0.5, 0.3, 0.2),  (0.6, 0.2, 0.2)
    
]

# Custom Dataset with Dual Labels
from torch.utils.data import Dataset

class DualLabelDataset(Dataset):
    def __init__(self, root, transform=None):
        self.dataset = ImageFolder(root=root, transform=transform)
        self.transform = transform
        self.gender_labels = {0: 'African', 1: 'Asian', 2: 'Caucasian'}
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, class_idx = self.dataset[idx]
        gender_label = class_idx // 5  # Assuming 10 classes: 0-4 Male diseases, 5-9 Female diseases
        disease_label = class_idx % 5
        return image, disease_label, gender_label
    
    def get_by_gender_and_disease(self):
        gender_disease_dict = defaultdict(lambda: defaultdict(list))  # {gender: {disease: [samples]}}
        
        for idx in range(len(self.dataset)):
            image, class_idx = self.dataset[idx]
            gender_label = class_idx // 5
            disease_label = class_idx % 5
            gender_disease_dict[gender_label][disease_label].append(idx)
            
        return gender_disease_dict
    
    

def sample_by_gender_proportion(gender_disease_dict, num, gender_proportions):
    selected_indices = []
    
    total_proportion = sum(gender_proportions.values())
    gender_num_samples = {gender: int(num * gender_proportions[gender] / total_proportion) for gender in gender_proportions}
    
    for gender, num_samples in gender_num_samples.items():
        disease_dict = gender_disease_dict[gender]
        
        disease_count = len(disease_dict)
        samples_per_disease = num_samples // disease_count
        
        for disease, indices in disease_dict.items():
            if len(indices) >= samples_per_disease:
                selected_indices.extend(random.sample(indices, samples_per_disease))
            else:
                selected_indices.extend(indices) 

    return selected_indices




# Initialize the parser
parser = argparse.ArgumentParser(description="ViT Disease Classification with Fairness")
parser.add_argument("--data", type=str, required=True, help="Path to the original data directory")
parser.add_argument("--aug_data", type=str, default=None, help="Path to the augmented data directory")
parser.add_argument("--output", type=str, required=True, help="Path to the output directory for the model")
parser.add_argument("--method", type=str, choices=["ff", "lp"], default="ff", help="Method to use for training ('ff' for full fine-tuning, 'lp' for linear probing)")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
parser.add_argument("--batchsize", type=int, default=64, help="Batch size for training")
parser.add_argument("--seed", type=int, default=6, help="Random seed")



# Parse the arguments
args = parser.parse_args()
for seed in range(3,6,1):
    args.seed = seed
    for num in range(500,1000,100):
        if seed == 3:
            proportion_list2=[(0.2, 0.2, 0.6), (0.2, 0.3, 0.5), (0.2, 0.4, 0.4),
                              (0.2, 0.5, 0.3), (0.2, 0.6, 0.2), (0.333, 0.333, 0.333),
                              (0.3, 0.2, 0.5), (0.3, 0.3, 0.4), (0.3, 0.4, 0.3), (0.3, 0.5, 0.2),
                              (0.4, 0.2, 0.4), (0.4, 0.3, 0.3),
                              (0.4, 0.4, 0.2),  (0.5, 0.2, 0.3),
                              (0.5, 0.3, 0.2),  (0.6, 0.2, 0.2)
            ]
            for idx, proportions in enumerate(proportion_list2):
            #  race_proportions
                race_proportions = {0: proportions[0], 1: proportions[1], 2: proportions[2]}


                # Initialize wandb
                wandb.init(
                    project="final-skin-DPO",
                    name=f"{args.method}_lr{args.lr}_bs{args.batchsize}_aug{os.path.basename(args.aug_data) if args.aug_data else 'no_aug'}_trial{args.seed}_scale{race_proportions}_num{num}",
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

                # Load pre-trained ViT model with 5 output classes for disease classification
                model = ViTForImageClassification.from_pretrained(
                    'google/vit-base-patch16-224',
                )

                # Optionally, modify the classifier for better performance
                model.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(model.classifier.in_features, 5)
                )

                # Load feature extractor
                feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

                # Define transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
                ])

                # Initialize datasets
                train_dataset = DualLabelDataset(root=args.data, transform=transform)

                # Augmented dataset
                if args.aug_data:
                    aug_train_dataset = DualLabelDataset(root=args.aug_data, transform=transform)
                    race_disease_dict = aug_train_dataset.get_by_gender_and_disease()
                    # race_proportions = {0: 0.333, 1: 0.333, 2: 0.333}
                    selected_indices = sample_by_gender_proportion(race_disease_dict, num * 15, race_proportions)
                    selected_subset = Subset(aug_train_dataset, selected_indices)

                # Split the dataset into training and validation
                train_len = int(0.9 * len(train_dataset))
                val_len = len(train_dataset) - train_len
                train_subset, val_subset = random_split(train_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))

                # Augmented training subset
                if args.aug_data:
                    # aug_train_len = int(0.9 * len(aug_train_dataset))
                    # aug_train_len =  num
                    # aug_val_len = len(aug_train_dataset) - aug_train_len
                    # aug_train_subset, _ = random_split(aug_train_dataset, [aug_train_len, aug_val_len], generator=torch.Generator().manual_seed(args.seed))
                    combined_train_subset = ConcatDataset([train_subset, selected_subset])
                else:
                    combined_train_subset = train_subset

                # DataLoaders
                train_loader = DataLoader(
                    combined_train_subset,
                    batch_size=args.batchsize,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True 
                )

                val_loader = DataLoader(
                    val_subset,
                    batch_size=args.batchsize,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True 
                )

                # Set device
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)

                # Define optimizer and loss
                if args.method == "ff":
                    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
                elif args.method == "lp":
                    optimizer = AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=1e-5)
                else:
                    raise ValueError("Invalid method")

                scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
                criterion = CrossEntropyLoss()

                # Training Loop with Fairness Evaluation
                best_val_accuracy = 0.0
                patience = 3
                trigger_times = 0

                for epoch in range(args.epochs):
                    # Training
                    model.train()
                    running_loss = 0.0
                    for batch in train_loader:
                        inputs, disease_labels, _ = batch  # Ignore race labels during training
                        inputs = inputs.to(device)
                        disease_labels = disease_labels.to(device)
                        
                        outputs = model(pixel_values=inputs)
                        loss = criterion(outputs.logits, disease_labels)
                        
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
                            
                            outputs = model(pixel_values=inputs)
                            _, predicted = torch.max(outputs.logits, 1)
                            total += disease_labels.size(0)
                            correct += (predicted == disease_labels).sum().item()
                    
                    train_accuracy = 100 * correct / total
                    print(f'Train Accuracy: {train_accuracy:.2f}%')
                    
                    # Validation Accuracy and Fairness Metrics
                    race_correct = {0: 0, 1: 0, 2: 0}
                    race_total = {0: 0, 1: 0, 2: 0}
                    y_true = []
                    y_pred = []
                    race_labels = []
                    
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for batch_idx, (inputs, disease_labels, race_labels_batch) in enumerate(val_loader):
                            inputs = inputs.to(device)
                            disease_labels = disease_labels.to(device)
                            
                            outputs = model(pixel_values=inputs)
                            _, predicted = torch.max(outputs.logits, 1)
                            
                            total += disease_labels.size(0)
                            correct += (predicted == disease_labels).sum().item()
                            
                            # Collect labels for fairness metrics
                            y_true.extend(disease_labels.cpu().numpy())
                            y_pred.extend(predicted.cpu().numpy())
                            race_labels.extend(race_labels_batch.cpu().numpy())
                            
                            # Per-race accuracy
                            for i in range(len(disease_labels)):
                                race = race_labels_batch[i].item()
                                race_correct[race] += (predicted[i] == disease_labels[i]).item()
                                race_total[race] += 1
                    
                    overall_accuracy = 100 * correct / total
                    print(f'Validation Accuracy: {overall_accuracy:.2f}%')
                    
                    # Calculate and print accuracy for each race group
                    acc_list = []
                    for race in range(3):
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
                        "African_accuracy": acc_list[0],
                        "Asian_accuracy": acc_list[1],
                        "Caucasian_accuracy": acc_list[2]
                    })
                    
                    # Save the best model
                    # if overall_accuracy > best_val_accuracy:
                    #     best_val_accuracy = overall_accuracy
                    #     model.save_pretrained(os.path.join("model", args.output, "best_model"))
                    #     print("Best model saved!")
                    
                    # Early Stopping (Optional)
                    # Uncomment if you wish to implement early stopping
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
                    
                # Final Model Save
                # model.save_pretrained(os.path.join("model", args.output, "final_model"))
                wandb.finish()
                # wait_time = random.randint(0, 5) * 60 
                # time.sleep(wait_time)
                print("------------------------------")
        else:
            proportion_list2=[(0.2, 0.2, 0.6), (0.2, 0.3, 0.5), (0.2, 0.4, 0.4),
                              (0.2, 0.5, 0.3), (0.2, 0.6, 0.2), (0.333, 0.333, 0.333),
                              (0.3, 0.2, 0.5), (0.3, 0.3, 0.4), (0.3, 0.4, 0.3), (0.3, 0.5, 0.2),
                              (0.4, 0.2, 0.4), (0.4, 0.3, 0.3),
                              (0.4, 0.4, 0.2),  (0.5, 0.2, 0.3),
                              (0.5, 0.3, 0.2),  (0.6, 0.2, 0.2)
            ]
            for idx, proportions in enumerate(proportion_list2):
        #  race_proportions
                race_proportions = {0: proportions[0], 1: proportions[1], 2: proportions[2]}


                # Initialize wandb
                wandb.init(
                    project="final-skin-DPO",
                    # entity="RuichenZhang",
                    name=f"{args.method}_lr{args.lr}_bs{args.batchsize}_aug{os.path.basename(args.aug_data) if args.aug_data else 'no_aug'}_trial{args.seed}_scale{race_proportions}_num{num}",
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

                # Load pre-trained ViT model with 5 output classes for disease classification
                model = ViTForImageClassification.from_pretrained(
                    'google/vit-base-patch16-224',
                )

                # Optionally, modify the classifier for better performance
                model.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(model.classifier.in_features, 5)
                )

                # Load feature extractor
                feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

                # Define transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
                ])

                # Initialize datasets
                train_dataset = DualLabelDataset(root=args.data, transform=transform)

                # Augmented dataset
                if args.aug_data:
                    aug_train_dataset = DualLabelDataset(root=args.aug_data, transform=transform)
                    race_disease_dict = aug_train_dataset.get_by_gender_and_disease()
                    # race_proportions = {0: 0.333, 1: 0.333, 2: 0.333}
                    selected_indices = sample_by_gender_proportion(race_disease_dict, num * 15, race_proportions)
                    selected_subset = Subset(aug_train_dataset, selected_indices)

                # Split the dataset into training and validation
                train_len = int(0.9 * len(train_dataset))
                val_len = len(train_dataset) - train_len
                train_subset, val_subset = random_split(train_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))

                # Augmented training subset
                if args.aug_data:
                    # aug_train_len = int(0.9 * len(aug_train_dataset))
                    # aug_train_len =  num
                    # aug_val_len = len(aug_train_dataset) - aug_train_len
                    # aug_train_subset, _ = random_split(aug_train_dataset, [aug_train_len, aug_val_len], generator=torch.Generator().manual_seed(args.seed))
                    combined_train_subset = ConcatDataset([train_subset, selected_subset])
                else:
                    combined_train_subset = train_subset

                # DataLoaders
                train_loader = DataLoader(
                    combined_train_subset,
                    batch_size=args.batchsize,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True 
                )

                val_loader = DataLoader(
                    val_subset,
                    batch_size=args.batchsize,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True 
                )

                # Set device
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)

                # Define optimizer and loss
                if args.method == "ff":
                    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
                elif args.method == "lp":
                    optimizer = AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=1e-5)
                else:
                    raise ValueError("Invalid method")

                scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
                criterion = CrossEntropyLoss()

                # Training Loop with Fairness Evaluation
                best_val_accuracy = 0.0
                patience = 3
                trigger_times = 0

                for epoch in range(args.epochs):
                    # Training
                    model.train()
                    running_loss = 0.0
                    for batch in train_loader:
                        inputs, disease_labels, _ = batch  # Ignore race labels during training
                        inputs = inputs.to(device)
                        disease_labels = disease_labels.to(device)
                        
                        outputs = model(pixel_values=inputs)
                        loss = criterion(outputs.logits, disease_labels)
                        
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
                            
                            outputs = model(pixel_values=inputs)
                            _, predicted = torch.max(outputs.logits, 1)
                            total += disease_labels.size(0)
                            correct += (predicted == disease_labels).sum().item()
                    
                    train_accuracy = 100 * correct / total
                    print(f'Train Accuracy: {train_accuracy:.2f}%')
                    
                    # Validation Accuracy and Fairness Metrics
                    race_correct = {0: 0, 1: 0, 2: 0}
                    race_total = {0: 0, 1: 0, 2: 0}
                    y_true = []
                    y_pred = []
                    race_labels = []
                    
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for batch_idx, (inputs, disease_labels, race_labels_batch) in enumerate(val_loader):
                            inputs = inputs.to(device)
                            disease_labels = disease_labels.to(device)
                            
                            outputs = model(pixel_values=inputs)
                            _, predicted = torch.max(outputs.logits, 1)
                            
                            total += disease_labels.size(0)
                            correct += (predicted == disease_labels).sum().item()
                            
                            # Collect labels for fairness metrics
                            y_true.extend(disease_labels.cpu().numpy())
                            y_pred.extend(predicted.cpu().numpy())
                            race_labels.extend(race_labels_batch.cpu().numpy())
                            
                            # Per-race accuracy
                            for i in range(len(disease_labels)):
                                race = race_labels_batch[i].item()
                                race_correct[race] += (predicted[i] == disease_labels[i]).item()
                                race_total[race] += 1
                    
                    overall_accuracy = 100 * correct / total
                    print(f'Validation Accuracy: {overall_accuracy:.2f}%')
                    
                    # Calculate and print accuracy for each race group
                    acc_list = []
                    for race in range(3):
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
                        "African_accuracy": acc_list[0],
                        "Asian_accuracy": acc_list[1],
                        "Caucasian_accuracy": acc_list[2]
                    })
                    
                    # Save the best model
                    # if overall_accuracy > best_val_accuracy:
                    #     best_val_accuracy = overall_accuracy
                    #     model.save_pretrained(os.path.join("model", args.output, "best_model"))
                    #     print("Best model saved!")
                    
                    # Early Stopping (Optional)
                    # Uncomment if you wish to implement early stopping
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
                    
                # Final Model Save
                # model.save_pretrained(os.path.join("model", args.output, "final_model"))
                wandb.finish()
                # wait_time = random.randint(0, 5) * 60 
                # time.sleep(wait_time)
                print("------------------------------")
            