from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from tqdm.auto import tqdm

class BertFeatureExtractor:
    """
    A class for extracting BERT embeddings from pre-chunked text data.
    
    This class extracts various types of embeddings from the BERT model, including:
    - Mean pooled embedding (average of token embeddings)
    - The [CLS] token embedding (first token)
    - The pooler output (used typically for classification tasks)
    
    The embeddings can be extracted for each paragraph, where the input text is assumed 
    to be preprocessed and chunked into individual paragraphs stored in the "Essay" column
    of a pandas DataFrame.

    Attributes:
    ----------
    device : torch.device
        The device (CPU or CUDA) to run the model on.
    bert_tokenizer : BertTokenizer
        The BERT tokenizer to preprocess the input text.
    bert_model : BertModel
        The pre-trained BERT model to generate embeddings.
    """

    def __init__(self):
        """
        Initializes the BertFeatureExtractor with pre-loaded BERT model and tokenizer.

        The model and tokenizer are loaded from the 'bert-base-uncased' pre-trained model.
        The device is set to CUDA if a GPU is available, otherwise, it defaults to CPU.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)

    def get_bert_embedding(self, text, embedding_type='mean'):
        """
        Extracts BERT embeddings from a given text (paragraph) based on the specified embedding type.

        This method generates embeddings using BERT and returns one of the following types of embeddings:
        - 'mean': The mean-pooled embedding, computed by averaging the token embeddings.
        - '[CLS]': The embedding of the [CLS] token (the first token in the input).
        - 'pooler_output': The output from the pooler layer, typically used for classification tasks.
        
        Parameters:
        ----------
        text : str
            A single paragraph from the input data (assumed to be preprocessed).
        embedding_type : str, optional
            The type of embedding to extract. Options are 'mean', '[CLS]', and 'pooler_output'.
            Default is 'mean'.

        Returns:
        -------
        np.ndarray
            A numpy array containing the selected embedding for the paragraph.

        Raises:
        ------
        ValueError
            If the embedding_type is not one of 'mean', '[CLS]', or 'pooler_output'.
        """
        encoded_input = self.bert_tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True).to(self.device)
        with torch.no_grad():
            output = self.bert_model(**encoded_input)
        
        if embedding_type == 'mean':
            # Mean-pooling across all tokens (excluding padding)
            embedding = output.last_hidden_state.mean(1).cpu().numpy()
        elif embedding_type == '[CLS]':
            # [CLS] token embedding (first token in the sequence)
            embedding = output.last_hidden_state[:, 0, :].cpu().numpy()
        elif embedding_type == 'pooler_output':
            # Pooler output (used for classification tasks)
            embedding = output.pooler_output.cpu().numpy()
        else:
            raise ValueError(f"Unsupported embedding type '{embedding_type}'. Choose from 'mean', '[CLS]', or 'pooler_output'.")
        
        return embedding

    def transform(self, df, embedding_type='mean'):
        """
        Extracts BERT embeddings of the specified type for each paragraph in the preprocessed DataFrame.
        Returns a list of numpy arrays of embeddings for each paragraph.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame containing the chunked text data in the 'Essay' column.
        embedding_type : str, optional
            The type of embedding to extract. Options are 'mean', '[CLS]', and 'pooler_output'.
            Default is 'mean'.

        Returns:
        -------
        list of np.ndarray
            A list of numpy arrays containing the embeddings for each paragraph in the dataset.
        """
        embeddings = []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            text = row['essay']  # Paragraph is stored in the 'Essay' column

            # Extract embedding for the paragraph
            embedding = self.get_bert_embedding(text, embedding_type)
            embeddings.append(embedding)  # Store only the paragraph embedding

        return np.array(embeddings).squeeze(axis=1)
