from collections import Counter, defaultdict


class NaiveBayesClassifier:
	'''
	'''

	def __init__(self, alpha: float = 1.0):
		''' '''
		self.alpha = alpha
		self.log_priors = {}
		self.log_likelihoods = defaultdict(dict)
		self.classes = []
		self.vocab = set()

	def fit(self, X_train: List[str], y_train: List[int]):
		n = len(X_train)
		self.classes = np.unique(y_train)

		class_word_counts = {cls: Counter() for cls in self.classes}

		for text, label in zip(X_train, y_train):
			tokens = self._tokenize(text)
			class_word_counts[label].update(tokens)
			self.vocab.update(tokens)

		vocab_size = len(self.vocab)

		for cls in self.classes:
			self.log_priors[cls] = np.log(np.sum(np.array(y_train) == cls) / n)
			num_tokens_in_class = sum(class_word_counts[cls].values())
			denominator = num_tokens_in_class + self.alpha * vocab_size

			for word in self.vocab:
				word_count = class_word_counts[cls].get(word, 0)
				numerator = word_count + self.alpha
				self.log_likelihoods[cls][word] = np.log(numerator / denominator)
			self.log_likelihoods[cls]['<unseen>'] = np.log(self.alpha / denominator)


	def predict(self, X_test: List[str]) -> np.ndarray:
		''' '''
		predictions = np.array([self._predict_single(text) for text in X_test])
		return predictions

	def _predict_single(self, text: str) -> int:
		tokens = self._tokenize(text)

		predictions = {}
		for cls in self.classes:
			predictions[cls] = self.log_priors[cls] + sum([self.log_likelihoods[cls].get(token, self.log_likelihoods[cls]['<unseen>']) for token in tokens])

		return max(predictions, key = predictions.get)

	def _tokenize(self, text: str) -> List[str]:
		return re.findall(r'/b/w+/b', text.lower())