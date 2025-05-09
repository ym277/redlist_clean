import torch
import torch.nn as nn
from transformers import BertModel

class RedlistPredictor(nn.Module):
    def __init__(
        self,
        bert_model_name: str,
        structured_input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 6,
        share_bert: bool = False
    ):
        super().__init__()

        # Main BERT encoder for descriptive text
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Optional: share BERT weights between main text and taxonomy
        if share_bert:
            self.tax_bert = self.bert
        else:
            self.tax_bert = BertModel.from_pretrained(bert_model_name)

        # Projection heads
        self.text_proj = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        self.tax_proj = nn.Linear(self.tax_bert.config.hidden_size, hidden_dim)
        self.struct_proj = nn.Linear(structured_input_dim, hidden_dim)

        # Final classifier
        self.combined_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        tax_input_ids,
        tax_attention_mask,
        structured_input
    ):
        # Main description text
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_embed = self.text_proj(text_outputs.pooler_output)  # [B, hidden_dim]

        # Taxonomy text
        tax_outputs = self.tax_bert(input_ids=tax_input_ids, attention_mask=tax_attention_mask)
        tax_embed = self.tax_proj(tax_outputs.pooler_output)  # [B, hidden_dim]

        # Structured features
        struct_embed = self.struct_proj(structured_input)  # [B, hidden_dim]

        # Combine all
        combined = torch.cat((text_embed, tax_embed, struct_embed), dim=1)  # [B, 3*hidden_dim]
        logits = self.combined_proj(combined)

        return logits
