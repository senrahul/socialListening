import re
import logging
import fasttext
import configuration as config

logger = logging.getLogger(__name__)
PROBABILTIY_THRESHOLD = 0.5

class Classifier:
	def __init__(self):
		self._model_file = config.model_file_path
		self._model = None
		self._load_model()

	def _load_model(self):
		try:
			self._model = fasttext.load_model(self._model_file)
		except Exception as e:
			logger.error("MODEL UPLOAD FAILED ....%s"+str(e))

	def _parse_comment(self, comment):
		comment_encoded_ascii = comment.encode("ascii", errors="ignore").decode()
		comment_trimmed = re.sub("\s+", " ", comment_encoded_ascii.strip().lower())
		return comment_trimmed

	def _predict_category_list(self, comment):
		cum_prob = 0
		candidate_dimensions = []
		if self._model:	
			predictions = self._model.predict_proba([comment], k = 3)[0]
			for predicted_category, probability in predictions:
				dimensionid = predicted_category.replace("__label__", "")
				if cum_prob < PROBABILTIY_THRESHOLD:
					cum_prob += probability
					candidate_dimensions.append((str(dimensionid), str(probability)))
		else:
			logger.warn("MODEL NOT LOADED : RETURNING NONE PREDICTION FOR QUERY %s"%comment)
		return candidate_dimensions

	def predict(self, comment):
		'''returns list of tuples'
		element 1 = category
		element 2 = probabiltiy
		'''
		comment_clean = self._parse_comment(comment)
		predicted_categories = self._predict_category_list(comment_clean)
		return predicted_categories


