# bert_regressor.py
from transformers import BertModel
from torch.utils.data import Dataset
import torch
from torch import nn

# Dataset Class
class EssayDataset(Dataset):
    def __init__(self, essays, scores, tokenizer, max_len=512):
        self.essays = essays
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.essays)

    def __getitem__(self, index):
        essay = str(self.essays[index])
        score = self.scores[index]
        encoding = self.tokenizer(
            essay,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'score': torch.tensor(score, dtype=torch.float)
        }

# Model Class
class BertRegressor(nn.Module):
    def __init__(self):
        super(BertRegressor, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        x = torch.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)
