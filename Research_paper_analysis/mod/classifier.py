from sentence_transformers import SentenceTransformer
from .text_processing import extract_features_from_pdf
import numpy as np
from .utils import find_n_closest_vectors, most_frequent_element
from .text_processing import extract_abstract
from .LLM import get_rationale_from_LLM

def predict_paper_publishability(pdf_path, length_range=[0,160], density_range=[0,1.8], coherence_range=[0.40, 1], include_coherence=True):
    """
    Predicts whether a research paper is publishable based on its textual features.

    The function evaluates the paper's sentence length, mathematical/numerical density, 
    and optionally coherence score, against predefined thresholds.

    Args:
        pdf_path (str): Path to the research paper PDF.
        length_range (list, optional): Allowed range for average sentence length. Defaults to [0, 160].
        density_range (list, optional): Allowed range for mathematical/numerical character density. Defaults to [0, 1.8].
        coherence_range (list, optional): Allowed range for coherence score. Defaults to [0.40, 1].
        include_coherence (bool, optional): Whether to include coherence in the evaluation. Defaults to True.

    Returns:
        bool: True if the paper meets the criteria for publishability, otherwise False.
    """

    if include_coherence:
        sentence_length, density, coherence = extract_features_from_pdf(pdf_path, include_coherence=include_coherence)
        if (length_range[0]<=sentence_length<=length_range[1]) and (density_range[0]<=density<=density_range[1]) and (coherence_range[0]<=coherence<=coherence_range[1]):
            return True
        return False 
    else:
        sentence_length, density = extract_features_from_pdf(pdf_path, include_coherence=False)
        if (length_range[0]<=sentence_length<=length_range[1]) and (density_range[0]<=density<=density_range[1]):
            return True
        return False   
    
def predict_conference_reasoning(pdf_path, api_key): # this is the final funciton that gives the predicted conference of the research paper pdf
    """
    Predicts the most suitable academic conference for a research paper and provides reasoning.

    This function extracts the abstract from the given PDF, encodes it using a Sentence Transformer model, 
    and finds the closest matching training vectors to determine the conference. 
    It then generates a rationale for the classification using an LLM.

    Args:
        pdf_path (str): Path to the research paper PDF.
        api_key (str): API key for querying the LLM to generate reasoning.

    Returns:
        tuple:
            - str: Predicted conference (one of 'CVPR', 'EMNLP', 'KDD', 'NeurIPS', 'TMLR').
            - str: Explanation from the LLM on why the paper belongs to the predicted conference.
    """

    titles_vector_base = np.load('TrainData/KNN_train_data.npy') # Vector store can be used here
    titles_vector_labels = np.load('TrainData/KNN_train_labels.npy')
    class_label = ['CVPR', 'EMNLP', 'KDD', 'NeurIPS', 'TMLR']
    abstract = extract_abstract(pdf_path)
    model_name = 'all-mpnet-base-v2'
    model = SentenceTransformer(model_name)
    abstract_embed = model.encode(abstract)
    index = find_n_closest_vectors(titles_vector_base, abstract_embed, n=21)
    l = [titles_vector_labels[i] for i in index]
    conference_class = most_frequent_element(l)
    conference = class_label[conference_class]
    rationale = get_rationale_from_LLM(abstract, conference, api_key)
    return conference, rationale