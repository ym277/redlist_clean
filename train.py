import pandas as pd
import numpy as np
import ast
from transformers import BertTokenizer
import pickle as pkl
from dataset import RedlistDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from transformers import AdamW
from model import RedlistPredictor
from tqdm import tqdm
from transformers import get_scheduler
import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sys

num_epochs = 6

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    print('-- begin evaluation')

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tax_input_ids = batch['tax_input_ids'].to(device)
            tax_attention_mask = batch['tax_attention_mask'].to(device)
            structured_input = batch['structured_input'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask, tax_input_ids, tax_attention_mask, structured_input)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    print('-- Evaluation results:')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Loss: {avg_loss:.4f}")

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }, all_preds

def train_one_epoch_with_progress(model, loader, optimizer, criterion, scheduler=None):
    model.train()
    total_loss = 0
    all_losses = []
    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tax_input_ids = batch['tax_input_ids'].to(device)
        tax_attention_mask = batch['tax_attention_mask'].to(device)
        structured_input = batch['structured_input'].to(device)
        labels = batch['label'].to(device)

        logits = model(input_ids, attention_mask, tax_input_ids, tax_attention_mask, structured_input)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item()
        all_losses.append(loss.item())

        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx + 1}/{len(loader)} - Loss: {loss.item():.4f}")

    return total_loss / len(loader), all_losses

if __name__ == "__main__":
    job_name = sys.argv[1]

    with open("data/data_splits.pkl", "rb") as f:
        data = pkl.load(f)
        df_train = data["train"]
        df_test = data["test"]
        df_deficient = data["deficient"]

    if job_name == 'no_redact' or job_name == 'no_redact_try':
        label_to_category = {
            0: 'Least Concern',
            1: 'Near Threatened',
            2: 'Vulnerable',
            3: 'Endangered',
            4: 'Critically Endangered',
            5: 'Extinct'
        }

        print('beging updating redacted labels')
        tqdm.pandas()
        df_train['rationale'] = df_train.progress_apply(
            lambda row: row['rationale'].replace('[REDACTED]', label_to_category[row['redlistCategory'] - 1]),
            axis=1
        )
        print('finished updating training set')
        df_test['rationale'] = df_test.progress_apply(
            lambda row: row['rationale'].replace('[REDACTED]', label_to_category[row['redlistCategory'] - 1]),
            axis=1
        )
        print('finished updating test set')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = RedlistDataset(df_train, tokenizer)
    test_dataset = RedlistDataset(df_test, tokenizer)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    print('finished loading data')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RedlistPredictor(
        bert_model_name='bert-base-uncased',
        structured_input_dim=len(df_train.iloc[0]['structured_vector']),
        hidden_dim=256,
        num_classes=6
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    total_steps = len(train_loader) * num_epochs

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    checkpoint_dir = f"checkpoints_{job_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_losses = []

    print('starting training, num_epochs:', num_epochs)

    for epoch in range(4, num_epochs+1):
        print(f"Epoch {epoch}/{num_epochs}")
        train_loss, losses_epoch = train_one_epoch_with_progress(model, train_loader, optimizer, criterion, scheduler)
        test_res, test_preds = evaluate(model, test_loader, criterion)
        # val_loss = val_res['loss']
        train_losses.append(train_loss)

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test Loss:   {test_res['loss']:.4f}")
        # print(f"  Val Acc:    {test_res['accuracy']:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_results': test_res,
            'test_preds': test_preds,
        }
        torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pt")
