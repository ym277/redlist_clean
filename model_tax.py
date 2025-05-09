import torch
import torch.nn as nn
from transformers import BertModel

class RedlistPredictor(nn.Module):
    def __init__(
        self,
        bert_model_name: str,
        structured_input_dim: int,
        taxonomy_vocab_sizes: dict,  # {'className': 40, 'orderName': 100, ...}
        taxo_embed_dim: int = 8,
        hidden_dim: int = 256,
        num_classes: int = 6
    ):
        super().__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)
        self.text_proj = nn.Linear(self.bert.config.hidden_size, hidden_dim)

        # Taxonomy embeddings
        self.taxo_embeddings = nn.ModuleDict({
            level: nn.Embedding(num_embeddings=vocab_size, embedding_dim=taxo_embed_dim)
            for level, vocab_size in taxonomy_vocab_sizes.items()
        })

        # Update structured input dimension to include taxonomy
        total_taxo_dim = taxo_embed_dim * len(taxonomy_vocab_sizes)
        self.struct_proj = nn.Linear(structured_input_dim + total_taxo_dim, hidden_dim)

        self.combined_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask, structured_input, taxonomy=None):
        # BERT text encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        text_embed = self.text_proj(pooled_output)
        # text_embed = 0
        # print(self.taxo_embeddings)

        # Taxonomy embedding lookup and concat
        taxo_embeds = [
            self.taxo_embeddings[level](taxonomy[:, i]) for i, level in enumerate(self.taxo_embeddings.keys())
        ]
        taxo_vector = torch.cat(taxo_embeds, dim=1)  # [batch_size, total_taxo_dim]

        # Structured + taxonomy
        full_structured = torch.cat((structured_input, taxo_vector), dim=1)
        struct_embed = self.struct_proj(full_structured)

        # Final combined
        combined = torch.cat((text_embed, struct_embed), dim=1)
        logits = self.combined_proj(combined)

        return logits
