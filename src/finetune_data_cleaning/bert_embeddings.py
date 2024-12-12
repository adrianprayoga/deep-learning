# copied from different project

from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from tqdm.auto import tqdm

class BertFeatureExtractor:
    """
    Extracts mean embeddings for 
    1) nsfw prediction
    2) contains_person prediction
    """

    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)

    def get_mean_embedding(self, text):

        encoded_input = self.bert_tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True).to(self.device)
        with torch.no_grad():
            output = self.bert_model(**encoded_input)
        embedding = output.last_hidden_state.mean(1).cpu().numpy()
        return embedding

    def transform(self, df, text_column='prompt'):
        """
        gets mean-pooled BERT embeddings for column of a DataFrame.
        """
        embeddings = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            text = row[text_column]
            embedding = self.get_mean_embedding(text)
            embeddings.append(embedding)
        return np.array(embeddings).squeeze(axis=1)
