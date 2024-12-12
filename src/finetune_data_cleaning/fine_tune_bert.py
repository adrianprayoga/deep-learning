# copied from different project

# bert_regressors.py
from transformers import BertModel
from torch.utils.data import Dataset
import torch
from torch import nn

class ImagePromptDataset(Dataset):
    """
    Dataset for image prompt tasks, containing essays, and two target variables:
    'contains_person' and 'nsfw_score'.
    """
    def __init__(self, essays, contains_person, nsfw_scores, tokenizer, max_len=512):
        self.essays = essays
        self.contains_person = contains_person
        self.nsfw_scores = nsfw_scores
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.essays)

    def __getitem__(self, index):
        essay = str(self.essays[index])
        contains_person = self.contains_person[index]
        nsfw_score = self.nsfw_scores[index]
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
            'contains_person': torch.tensor(contains_person, dtype=torch.float),
            'nsfw_score': torch.tensor(nsfw_score, dtype=torch.float)
        }

class BertRegressors(nn.Module):
    """
    BERT-based dual regressor model for predicting:
    1. Whether a prompt corresponds to a picture containing a person ('contains_person').
    2. The probability that a picture is NSFW ('nsfw_score').
    """
    def __init__(self):
        super(BertRegressors, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p=0.3)
        self.fc_contains_person = nn.Linear(self.bert.config.hidden_size, 1)
        self.fc_nsfw = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        contains_person_pred = torch.sigmoid(self.fc_contains_person(pooled_output))
        nsfw_pred = torch.sigmoid(self.fc_nsfw(pooled_output))
        return contains_person_pred.squeeze(-1), nsfw_pred.squeeze(-1)
