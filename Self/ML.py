import numpy as np
from collections import Counter, defaultdict
import re

class NaiveBayesClassifier:
    """
    A Naive Bayes Classifier for text classification with Laplace smoothing.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initializes the classifier.
        
        Args:
            alpha (float): The smoothing parameter for Laplace smoothing.
        """
        self.alpha = alpha
        self.log_priors = {}
        self.log_likelihoods = defaultdict(dict)
        self.vocab = set()
        self.classes = []

    def fit(self, X_train: list[str], y_train: list[int]):
        """
        Trains the Naive Bayes model on the training data.

        Args:
            X_train (list[str]): A list of text documents.
            y_train (list[int]): A list of corresponding labels (e.g., 0 or 1).
        """
        # Get unique classes and total number of documents
        self.classes = np.unique(y_train)
        n_docs = len(X_train)
        
        # --- Step 1: Tokenize and Count Words ---
        # A dictionary to hold word counts for each class
        class_word_counts = {cls: Counter() for cls in self.classes}
        
        for text, label in zip(X_train, y_train):
            tokens = self._tokenize(text)
            class_word_counts[label].update(tokens)
            self.vocab.update(tokens)
            
        vocab_size = len(self.vocab)

        # --- Step 2: Calculate Priors and Likelihoods ---
        for cls in self.classes:
            # Calculate log prior for the class
            class_docs_count = np.sum(np.array(y_train) == cls)
            self.log_priors[cls] = np.log(class_docs_count / n_docs)
            
            # Calculate total words in the class
            total_words_in_class = sum(class_word_counts[cls].values())
            
            # Calculate log likelihoods for each word in the vocabulary
            denominator = total_words_in_class + self.alpha * vocab_size
            
            for word in self.vocab:
                word_count = class_word_counts[cls].get(word, 0)
                numerator = word_count + self.alpha
                self.log_likelihoods[cls][word] = np.log(numerator / denominator)
                
            # Create a likelihood for unseen words during prediction
            self.log_likelihoods[cls]['<unseen>'] = np.log(self.alpha / denominator)

    def predict(self, X_test: list[str]) -> np.ndarray:
        """
        Makes predictions on a list of new documents.

        Args:
            X_test (list[str]): A list of text documents to classify.
        
        Returns:
            np.ndarray: An array of predicted labels.
        """
        predictions = [self._predict_single(text) for text in X_test]
        return np.array(predictions)

    def _predict_single(self, text: str) -> int:
        """Helper function to classify a single document."""
        tokens = self._tokenize(text)
        
        # Calculate the posterior log probability for each class
        posterior_log_probs = {}
        for cls in self.classes:
            # Start with the prior
            log_prob = self.log_priors[cls]
            
            # Add the likelihoods for each word in the document
            for token in tokens:
                # Use the pre-calculated log likelihood. If word is unseen, use the special token.
                log_prob += self.log_likelihoods[cls].get(token, self.log_likelihoods[cls]['<unseen>'])
            
            posterior_log_probs[cls] = log_prob
            
        # Return the class with the highest log probability
        return max(posterior_log_probs, key=posterior_log_probs.get)

    def _tokenize(self, text: str) -> list[str]:
        """A simple tokenizer to convert text to a list of words."""
        # Convert to lowercase and find all word sequences
        return re.findall(r'\b\w+\b', text.lower())

