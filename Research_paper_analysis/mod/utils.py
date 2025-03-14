from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import statistics
from transformers import BertTokenizer
from transformers import BertModel
from collections import Counter


def get_coherence(s, tokenizer, model):
    """
    Compute the coherence score of a given sentence using a BERT model.

    Args:
        s (str): Input sentence.
        tokenizer (BertTokenizer): Pre-trained BERT tokenizer.
        model (BertModel): Pre-trained BERT model.

    Returns:
        float: The average pairwise cosine similarity of token embeddings (excluding [CLS] and [SEP]).
    """

    sentence1 = s
    tokens1 = tokenizer(sentence1, return_tensors="pt", padding=True, truncation=True)
    outputs1 = model(**tokens1)

    # Extract token embeddings (excluding [CLS] and [SEP])
    token_embeddings1 = outputs1.last_hidden_state.squeeze(0)[1:-1]

    # Compute pairwise cosine similarity for tokens
    similarity_matrix1 = cosine_similarity(token_embeddings1.detach().numpy())

    coherence_score1 = similarity_matrix1.mean()

    return coherence_score1

def get_coherence_list(sent):  
    """
    Compute coherence scores for a list of sentences using BERT.

    Args:
        sent (list of str): List of sentences.

    Returns:
        list of float: List of coherence scores for each sentence.
    """
    path1_coherence = []
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    for i in sent:
        path1_coherence.append(get_coherence(i, tokenizer, model))
    return path1_coherence

def get_sent_len_list(sent):
    """
    Compute the length of each sentence in a given list.

    Args:
        sent (list of str): List of sentences.

    Returns:
        list of int: List of sentence lengths.
    """

    return [len(i) for i in sent]

def find_n_closest_vectors(training_vectors, new_vector, n=1):
    """
    Find the indices of the n closest vectors to the new vector among the training vectors.
    
    Args:
        training_vectors (list or np.ndarray): List or array of training vectors.
        new_vector (np.ndarray): The new vector to compare.
        n (int): Number of closest vectors to find.
    
    Returns:
        list: Indices of the n closest vectors in the training vectors.
    """
    training_vectors = np.array(training_vectors)  # Ensure it's a NumPy array
    new_vector = np.array(new_vector).reshape(1, -1)  # Reshape to match dimensions
    similarities = cosine_similarity(training_vectors, new_vector).flatten()  # Compute cosine similarities
    closest_indices = np.argsort(similarities)[-n:][::-1]  # Get indices of n highest similarities in descending order
    return closest_indices.tolist()


def most_frequent_element(lst):
    """
    Find the element with the highest frequency of occurrence in a list.
    
    Args:
        lst (list): Input list of elements.
    
    Returns:
        The element with the highest frequency.
    """
    if not lst:
        return None  # Handle empty list case
    
    counter = Counter(lst)  # Count the frequency of each element
    most_common_element = counter.most_common(1)[0][0]  # Get the element with the highest frequency
    return most_common_element