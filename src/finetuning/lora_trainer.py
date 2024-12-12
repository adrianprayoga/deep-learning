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
            probabilities = torch.softmax(outputs, dim=1)[:, 1]  
            y_pred.extend(probabilities.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    return y_pred, y_true

from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_loop(train_dataset_fp, dataset_size, use_lora):
    config = TrainingConfig()
    config.train_dataset = train_dataset_fp
    config.use_lora = use_lora
    config.dataset_size = dataset_size
    config.update_hyperparams()

    torch.manual_seed(config.seed)


    train_dataset = TrainingDataset(config=config)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True
    )


    val_loaders = {
        i: DataLoader(
            InferenceDataset(config=config, dataset_index=i),
            batch_size=config.batch_size, shuffle=False, num_workers=3, pin_memory=True
        )
        for i in range(1, 4)
    }

    model = CLIPFineTuneDetector(config=config).to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)

    for epoch in range(config.epochs):
        config.current_epoch = epoch


        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config)
        config.loss = train_loss
        print(f"Epoch {epoch+1}/{config.epochs}, Training Loss: {train_loss:.4f}")

        for i, val_loader in val_loaders.items():
            dataset_name = getattr(config, f"val_dataset{i}_name")
            y_pred, y_true = validate(model, val_loader, config, dataset_name)
            metrics = get_test_metrics(y_pred, y_true, config, dataset_name, save=True)
            print(f"Validation on {dataset_name}: {metrics}")


            scheduler.step(metrics['auc'])

    print(f"Training completed for {train_dataset_fp} with size {dataset_size}.")

def eval_only():
    config = TrainingConfig()
    config.train_dataset = '/home/ginger/code/gderiddershanghai/deep-learning/data/MidJourney'
    config.use_lora = True
    config.dataset_size = 200
    config.update_hyperparams()

    torch.manual_seed(config.seed)


    train_dataset = TrainingDataset(config=config)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    fp1 = '/home/ginger/code/gderiddershanghai/deep-learning/data/CollabDiff'
    fp2 = '/home/ginger/code/gderiddershanghai/deep-learning/data/JDB_random'
    fp3 =  '/home/ginger/code/gderiddershanghai/deep-learning/data/JDB_train'
    
    config.val_dataset1 = fp1
    config.val_dataset1_name = 'CollabDiff'
    config.val_dataset2 = fp2
    config.val_dataset1_name = 'JDB_random'
    config.val_dataset3 = fp3
    config.val_dataset1_name = 'JDB_train'
    config.update_hyperparams()
    val_loaders = {
        i: DataLoader(
            InferenceDataset(config=config, dataset_index=i),
            batch_size=config.batch_size, shuffle=False, num_workers=3, pin_memory=True
        )
        for i in range(1, 4)
    }

    model = CLIPFineTuneDetector(config=config).to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)

    for i, val_loader in val_loaders.items():
        dataset_name = getattr(config, f"val_dataset{i}_name")
        y_pred, y_true = validate(model, val_loader, config, dataset_name)
        metrics = get_test_metrics(y_pred, y_true, config, dataset_name, save=True)
        print(f"Validation on {dataset_name}: {metrics}")

        scheduler.step(metrics['auc'])



def main():

    fp1 = '/home/ginger/code/gderiddershanghai/deep-learning/data/CollabDiff'
    fp2 = '/home/ginger/code/gderiddershanghai/deep-learning/data/JDB_random'
    fp3 =  '/home/ginger/code/gderiddershanghai/deep-learning/data/MidJourney'
    fp4 = 'mixed'
    # config = TrainingConfig()
    # config.train_dataset = 'mixed'
    
    # config.use_lora = True
    # config.dataset_size = 200
    # config.update_hyperparams()
    # print(config.csv_path)
    # for use_lora in [False]:
    #     for dataset_size in [200, 1000, 2000]:
    #         train_loop(train_dataset_fp=fp1, dataset_size=dataset_size, use_lora=use_lora)
    # for use_lora in [True]:
    #     for dataset_size in [500, 2000, 5000, 10000]:
    #         train_loop(train_dataset_fp=fp2, dataset_size=dataset_size, use_lora=use_lora)
    for use_lora in [True, False]:
        for dataset_size in [1000, 5000, 10000, 20000]:
            train_loop(train_dataset_fp=fp4, dataset_size=dataset_size, use_lora=use_lora)

if __name__ == "__main__":
    main()
    # eval_only()