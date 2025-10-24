import re
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from typing import List, Tuple

def set_length(bag_of_words: List[str], length: int) -> List[str]:
    '''Sets the size of each word in bag_of_words to the given 
        length by truncating or padding'''
    l = len(bag_of_words) 
    if l > length:
        # truncate the data
        return bag_of_words[:length]
    else:
        # pad the data
        return bag_of_words + ['<PAD>'] * (length - l)

def _load_and_process_reviews(path: Path, length: int) -> List[List[str]]:
    """Loads and preprocesses all review files from a given directory"""
    data = []
    for file_path in path.iterdir():
        with file_path.open(encoding='latin-1') as f:
            text = f.read()
            tokens = [word for word in re.findall(r'\b\w+\b', text.lower()) if word]
            data.append(set_length(tokens, length))
    return data

def preprocess(pos_path: str, neg_path: str, length: int = 250) -> Tuple[List[List[str]], np.ndarray]:
    """Preprocesses positive and negative review datasets"""
    pos_path = Path(pos_path)
    neg_path = Path(neg_path)

    pos_data = _load_and_process_reviews(pos_path, length)
    neg_data = _load_and_process_reviews(neg_path, length)

    data = pos_data + neg_data
    labels = np.array([1] * len(pos_data) + [0] * len(neg_data))
    
    return data, labels

def train(train_data: List[List[str]], train_label: List[int], k: int = 5, alpha: float = 1.0) -> Tuple[np.ndarray, List[dict]]:
    """Train the Naive Bayes Classifier"""
    n = len(train_data)

    # Count the words of each class separately
    class_word_count = [Counter(), Counter()] # 0th dict for neg class and 1st for pos class
    for data, label in zip(train_data, train_label):
        class_word_count[label].update(data)

    # Get the full vocabulary from train_data
    vocabulary = set(class_word_count[0].keys()) | set(class_word_count[1].keys())
    vocabulary_size = len(vocabulary)

    # Calculate class priors
    n_0 = (train_label == 0).sum()
    n_1 = n - n_0
    Py = np.array([n_0 / n, n_1 / n])

    # Calculate total word counts per class
    class_num_tokens = [sum(class_word_count[0].values()), sum(class_word_count[1].values())]
    
    # Calculate conditional probabilities
    Pxy = [{}, {}]
    for i in range(2):
        denominator = class_num_tokens[i] + alpha * vocabulary_size
        for word in vocabulary:
            word_count = class_word_count[i].get(word, 0)
            Pxy[i][word] = (word_count + alpha) / denominator
        # Probability for words not seen in the entiry training vocabulary
        Pxy[i]['unseen'] = alpha / denominator
                
    return Py, Pxy


def prediction(dataset, Py, Pxy):
    """
    Predicts the class of each data using Naive Bayes Classifier.

    dataset : preprocessed array of bag of words
    Py : prior probabilities
    Pxy : conditional probabilities
    """
    n = len(dataset)
    predicted_labels = []
    log_Py = np.log(Py)

    # use tqdm for a nice progress bar
    for data in tqdm(dataset, desc="Predicting"):
        log_probs = np.zeros(2)
        for i in range(2): # For each class (0 and 1)
            class_log_prob = sum(np.log(Pxy[i].get(word, Pxy[i]['unseen'])) for word in data)
            log_probs[i] = log_Py[i] + class_log_prob

        predicted_labels.append(np.argmax(log_probs))

    return np.array(predicted_labels)


# Main script execution
if __name__ == '__main__':
    DATA_DIR = Path('aclImdb')
    TRAIN_PATH = DATA_DIR / 'train'
    TEST_PATH = DATA_DIR / 'test'
    pos_train_path = TRAIN_PATH / 'pos'
    neg_train_path = TRAIN_PATH / 'neg'
    pos_test_path = TEST_PATH / 'pos'
    neg_test_path = TEST_PATH / 'neg'
    
    # Preprocessing, Training and Training accuracy
    print("Preprocessing training data...")
    train_data, train_label = preprocess(pos_train_path, neg_train_path, length = 250)

    print("Training the model...")
    Py, Pxy = train(train_data, train_label, alpha = 1.0)
    
    print("Evaluating on training data...")
    predicted_train_label = prediction(train_data, Py, Pxy)
    train_accuracy = np.mean(train_label == predicted_train_label)
    print(f'Train accuracy:, {train_accuracy:.4f}')

    print('-' * 20)

    # Testing the classifier
    print("Preprocessing test data...")
    test_data, test_label = preprocess(pos_test_path, neg_test_path, length = 250)

    print("Evaluating on test data...")
    predicted_test_label = prediction(test_data, Py, Pxy)
    test_accuracy = np.mean(test_label == predicted_test_label)

    print(f'Test accuracy:, {test_accuracy:.4f}')