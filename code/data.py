from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_data(data, max_words=10000, max_length=100):
    """
    Preprocess text data using the Keras Tokenizer.

    This function tokenizes words and ensures all documents have the same length by performing padding.

    Parameters:
    - data (list of str): The text data to be preprocessed. Each element should be a string representing a document.
    - max_words (int, optional): The maximum number of words to keep in the tokenizer vocabulary. Default is 10000.
    - max_length (int, optional): The maximum length of the sequences after padding. Default is 100.

    Returns:
    - sequences (numpy.ndarray): The preprocessed text data as a 2D numpy array of shape (num_documents, max_length).
      Each element represents a sequence of token indices.

    """
    # Build vocabulary from training text data
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(data)

    # Use tokenizer to convert text data to sequences
    sequences = tokenizer.texts_to_sequences(data)
    # Pad sequences to ensure uniform length
    sequences = pad_sequences(sequences, maxlen=max_length)

    return sequences
