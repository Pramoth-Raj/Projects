import re
import statistics
from nltk.tokenize import sent_tokenize
from PyPDF2 import PdfReader
from .utils import get_coherence_list, get_sent_len_list


def extract_features_from_pdf(pdf_path, include_coherence=True):
    """
    Extracts text-based features from a given PDF document, including sentence length, 
    proportion of numerical and mathematical characters, and optionally coherence score.

    Args:
        pdf_path (str): Path to the PDF file.
        include_coherence (bool, optional): Whether to compute coherence scores for the document. Defaults to True.

    Returns:
        tuple: 
            - float: Average sentence length.
            - float: Proportion of numerical and mathematical characters in the text.
            - float (optional): Average coherence score (if `include_coherence` is True).
    """
    
    # Load the PDF
    reader = PdfReader(pdf_path)
    full_text = ""

    # Extract text from each page
    for page in reader.pages:
        full_text += page.extract_text() + " "

    # Define patterns
    numerical_pattern = r'[0-9]'
    math_pattern = r'[+\-*/=^%()]'
    math_pattern = r'[σ∑∫π√∞Δθλ+\-=*/^<>%∂µˆΓαγδθλϵ(){}]'
    
    # Calculate character counts
    total_characters = len(full_text)  # Total characters, including spaces and newlines
    numerical_count = len(re.findall(numerical_pattern, full_text))  # Count numerical characters
    math_count = len(re.findall(math_pattern, full_text))  # Count mathematical characters

    # Basic cleaning to remove headings, equations, and unnecessary content
    cleaned_text = re.sub(r"(\n|\\n)+", " ", full_text)  # Remove newlines
    cleaned_text = re.sub(r"[^\w\s.,!?-]", "", cleaned_text)  # Remove special characters
    cleaned_text = re.sub(r"\b[A-Z]{2,}\b", "", cleaned_text)  # Remove headings (all-uppercase words)

    # Tokenize into sentences
    sentences = sent_tokenize(cleaned_text)

    # Filter out equations (e.g., containing "=" or numbers with operators)
    sentences = [
        sentence.strip()
        for sentence in sentences
        if not re.search(r"[=+\-*/^]", sentence) and len(re.findall(r"\d", sentence)) < len(sentence.split()) // 2
    ]

    if include_coherence:
        coherence = statistics.mean(get_coherence_list(sentences))

    sent_len = statistics.mean(get_sent_len_list(sentences))

    if include_coherence:
        return sent_len, (math_count+numerical_count)/total_characters, coherence
    else:
        return sent_len, (math_count+numerical_count)/total_characters


def extract_abstract(pdf_path):
    """
    Extracts the abstract from a given PDF document.

    The function identifies the abstract based on common structural markers:
    - Assumes the title is the text before '\nAbstract\n'.
    - Extracts the abstract from the text between '\nAbstract\n' and '1 Introduction'.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: The extracted abstract, or an empty string if the markers are not found.
    """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()


    # Extract abstract
    abstract_start = text.find("\nAbstract\n") + len("\nAbstract\n")
    abstract_end = text.find("1 Introduction")
    abstract = text[abstract_start:abstract_end].strip() if abstract_start != -1 and abstract_end != -1 else ""

    return abstract