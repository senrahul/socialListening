import os
import re
import logging
import configuration as config
import fasttext
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

LABEL_PREFIX = "__label__"
logger = logging.getLogger("sentiment_classification_model_training_job")
hdlr = logging.FileHandler(config.LOG_FILE_NAME)
formatter = logging.Formatter(config.LOG_FORMAT)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)


class ModelTrainer:
	def __init__(self):
		self.data_file_tsv = config.raw_training_data_file
		self.labeled_file = config.labeled_training_file
		self.model_file = config.model_file_path
		self.label_index_list = config.label_index_list

	def generated_labeled_data(self):
		'''read the google sheet (tab delimited) and create training file in fasttext file format'''
		logger.info("formatting raw google sheet data")
		with open(self.labeled_file, 'w+') as ofile:
			with open(self.data_file_tsv) as ifile:
				raw_data = [line.strip().split('\t') for line in ifile]
				max_label_index = max(self.label_index_list)
				for line in raw_data:
					if len(line) >= max_label_index + 1 and line[max_label_index] != '':
						#ignoring non ascii characters e.g. emoticons, hindi encoding etc.
						
						comment_encoded_ascii = line[8].encode("ascii", errors="ignore")
						comment_trimmed = re.sub("\s+", " ", comment_encoded_ascii.lower())
						comment = re.sub(r'[?|$|.|!|\'|"|:|\\|/]',r'', comment_trimmed)
						label_joined = "_".join([line[label_index] for label_index in self.label_index_list])
						label = LABEL_PREFIX + label_joined
						ofile.write(label + " " + comment)
						ofile.write("\n")

	def train_model(self):
		'''train model and save it locally'''
		logger.info("training sentiment classifier")
		model = fasttext.supervised(self.labeled_file, self.model_file, bucket= 2000000, word_ngrams = 2, minn =2, maxn = 3)

	def train(self):
		self.generated_labeled_data()
		self.train_model()

if __name__ == "__main__":
	model_trainer = ModelTrainer()
	model_trainer.train()







