import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from finetune_detectors import CLIPFineTuneDetector
from finetuning_metrics import get_test_metrics
from finetune_dataset_loader import TrainingDataset
from inference_dataloader import InferenceDataset
from training_config import TrainingConfig


def train_one_epoch(model, train_loader, optimizer, criterion, config):
    model.train()
    running_loss = 0.0
    for images, labels, _ in tqdm(train_loader, desc=f"Training Epoch {config.current_epoch+1}/{config.epochs}"):
        # Ensure images are in float32 and on the correct device
        images = images.to(config.device).float()
        labels = labels.to(config.device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)



def validate(model, val_loader, config, dataset_name):
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for images, labels, _, _ in tqdm(val_loader, desc=f"Validating on {dataset_name}"):
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Take probabilities for class 1
            y_pred.extend(probabilities.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    return y_pred, y_true


def main():
    # Load configuration
    for truth_val in [True, False]:
        
        for size in [500,2000,5000,15000]:
            config = TrainingConfig()
            config.use_lora  = truth_val
            config.dataset_size = size

            # Set random seed for reproducibility
            torch.manual_seed(config.seed)

            # Initialize datasets and loaders
            train_dataset = TrainingDataset(config=config)
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
            for images, labels, _ in train_loader:
                print(f"Train Image tensor dtype: {images.dtype}, expected: torch.float32")
                break
            
            val_dataset1 = InferenceDataset(config=config, use_val_set1=True)
            val_loader1 = DataLoader(val_dataset1, batch_size=config.batch_size, shuffle=False, num_workers=4)
            #  image_tensor, label, image_path, original_image
            for images, labels, _, _ in val_loader1:
                print(f"Val 1 Image tensor dtype: {images.dtype}, expected: torch.float32")
                break
            val_dataset2 = InferenceDataset(config=config, use_val_set1=False)
            val_loader2 = DataLoader(val_dataset2, batch_size=config.batch_size, shuffle=False, num_workers=4)
            for images, labels, _, _ in val_loader2:
                print(f"Val 2 Image tensor dtype: {images.dtype}, expected: torch.float32")
                break
            # Initialize model
            model = CLIPFineTuneDetector(config=config).to(config.device)

            # Set optimizer and loss function
            optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01 if not config.use_lora else 0)
            criterion = torch.nn.CrossEntropyLoss()
            
            best_val_metric = float('-inf')  # or float('inf') for metrics like loss
            patience = 3  # Stop training after `patience` epochs with no improvement
            patience_counter = 0  
                        
            print('loaded optimizer, model, criterion')
            # print('baseline score')
            # for val_loader, dataset_name in [(val_loader1, config.val_dataset1_name), (val_loader2, config.val_dataset2_name)]:
            #     y_pred, y_true = validate(model, val_loader, config, dataset_name)
            #     metrics = get_test_metrics(y_pred, y_true, config, dataset_name, save=True)
            #     print(f"Validation on {dataset_name}: {metrics}")
            # Training loop
            for epoch in range(config.epochs):
                config.current_epoch = epoch

                # Train for one epoch
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config)
                config.loss = train_loss
                print(f"Epoch {epoch+1}/{config.epochs}, Training Loss: {train_loss:.4f}")
                print('size of the training data', config.dataset_size)

                # Validate on the first dataset (val_loader1) and track AUC for early stopping
                y_pred_val1, y_true_val1 = validate(model, val_loader1, config, config.val_dataset1_name)
                metrics_val1 = get_test_metrics(y_pred_val1, y_true_val1, config, config.val_dataset1_name, save=True)
                print(f"Validation on {config.val_dataset1_name}: {metrics_val1}")

                # Track the best AUC on the first validation set for early stopping
                current_val_metric = metrics_val1["auc"]
                if current_val_metric > best_val_metric:
                    best_val_metric = current_val_metric
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"No improvement in validation AUC. Patience counter: {patience_counter}/{patience}")

                # Validate on the second dataset (val_loader2) for logging only
                y_pred_val2, y_true_val2 = validate(model, val_loader2, config, config.val_dataset2_name)
                metrics_val2 = get_test_metrics(y_pred_val2, y_true_val2, config, config.val_dataset2_name, save=True)
                print(f"Validation on {config.val_dataset2_name}: {metrics_val2}")

                # Early stopping condition
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}. Best validation AUC on {config.val_dataset1_name}: {best_val_metric:.4f}")
                    break

                # # Validate on both datasets
                # for val_loader, dataset_name in [(val_loader1, config.val_dataset1_name), (val_loader2, config.val_dataset2_name)]:
                #     y_pred, y_true = validate(model, val_loader, config, dataset_name)
                #     metrics = get_test_metrics(y_pred, y_true, config, dataset_name, save=True)
                #     print(f"Validation on {dataset_name}: {metrics}")
                #     current_val_metric = metrics["auc"] 
                    
                #     if current_val_metric > best_val_metric:
                #         best_val_metric = current_val_metric
                #         patience_counter = 0  
                #     else:
                #         patience_counter += 1
                # if patience_counter >= patience:
                #     print(f"Early stopping triggered at epoch {epoch+1}. Best validation AUC: {best_val_metric:.4f}")
                #     break
            # # Save model checkpoint
            # checkpoint_path = os.path.join(config.checkpoint_dir, f"{config.model_name}_epoch{epoch+1}.pth")
            # torch.save(model.state_dict(), checkpoint_path)
            # print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()

