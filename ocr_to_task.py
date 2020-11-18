## Pipeline Execution Script
# Requires: Python 3.6+
import argparse
import os
from data_reader_ocr import ocr_reader
from predict_tasks import predict_tasks_from_obj
from utils import *

def ocr_to_task(ocr_text, model, bert_vectorizer, tags_vectorizer, intents_label_encoder, logger=None):
    if logger is None:
        logging.basicConfig(filename="newfile.log", 
                    format='%(asctime)s %(message)s', 
                    filemode='w') 
        logger=logging.getLogger()
        logger.setLevel(logging.INFO)
    
    logger.info("Reading OCR file")
    images = ocr_reader(ocr_text)
    cards = predict_tasks_from_obj(images["text"], images["coordinates"], model, bert_vectorizer, tags_vectorizer, intents_label_encoder, logger)

    return cards

if __name__ == "__main__":

	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument("--folder", "-f", help="Folder where model is stored", type=str, required=True)
	parser.add_argument("--ocr_input", "-ocr", help="Path to OCR file", type=str, required=False, default="/ocr_log.txt")
	parser.add_argument("--add_cls", "-cls", help="whether to add CLS, SEP tokens before and after each token in the output files")

	args = parser.parse_args()
	ocr_file_in = args.ocr_input
	load_folder_path = args.folder
	add_cls = args.add_cls

	bert_vectorizer = BERTVectorizer(sess, bert_model_hub_path)
	model = JointBertModel.load(load_folder_path, sess)

	intents_label_encoder, _ = load_intents_label_encoder(load_folder_path)
	tags_vectorizer, _ = load_tags_vectorizer(load_folder_path)

	ocr_text = open(ocr_file_in, 'r', encoding='utf-8').read()
	cards = ocr_to_task(ocr_text, model, bert_vectorizer, tags_vectorizer, intents_label_encoder)

	print(cards)