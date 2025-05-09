from torch.utils.data import Dataset
import torch
import numpy as np

class RedlistDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512, tax_max_len=64):
        self.texts = df['text_input'].tolist()
        self.tax_texts = df['tax_text'].tolist()
        self.structured = np.stack(df['structured_vector'].values)
        self.labels = df['redlistCategory'].values
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tax_max_len = tax_max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Tokenize main text input
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        # Tokenize taxonomy text
        tax_encoding = self.tokenizer(
            self.tax_texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.tax_max_len,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'tax_input_ids': tax_encoding['input_ids'].squeeze(0),
            'tax_attention_mask': tax_encoding['attention_mask'].squeeze(0),
            'structured_input': torch.tensor(self.structured[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx] - 1, dtype=torch.long)  # convert to 0-based
        }

        return item
