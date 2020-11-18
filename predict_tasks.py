from Reader_data import Reader
from bert_vectorizer import BERTVectorizer
from joint_bert_model import JointBertModel
from itertools import chain

import argparse
import os
import tensorflow as tf
from sklearn import metrics
import json
import logging 

from nltk.translate.bleu_score import sentence_bleu
from task_title_generator import *
from tags_vectorizer import get_cls_sep_markers, TagsVectorizer, LabelEncoder
from utils import *

## Gives task cards
def get_gen_tasks_singleNER(model_files, input_text, jbm_inputs):
    inputs = [jbm_inputs.input_ids, jbm_inputs.input_mask, jbm_inputs.segment_ids, jbm_inputs.valid_positions]
    if model_files.model.model_params["use_layout_data"]:
        inputs.append(jbm_inputs.coordinates)
    if model_files.model.model_params["use_cls"]:
        inputs.append(jbm_inputs.cls_sep_markers)
    predicted_tags, predicted_intents = model_files.model.predict_slots_intent(inputs, model_files.tags_vectorizer, 
                                                                               model_files.intents_label_encoder)

    tasks_list = []
    for i in range(len(input_text)):
        entities = extract_entities(input_text[i], " ".join(predicted_tags[i]), jbm_inputs.coordinates[i])
        tasks_list.append(gen_task_card(entities, predicted_intents[i]))

    return tasks_list

def get_gen_tasks_multiNER(model_files, input_text, jbm_inputs):
    inputs = [jbm_inputs.input_ids, jbm_inputs.input_mask, jbm_inputs.segment_ids, jbm_inputs.valid_positions, jbm_inputs.coordinates]
    if sum(model_files.model.model_params["use_cls"]) > 0:
        inputs.append(jbm_inputs.cls_sep_markers)
    predicted_tags, predicted_intents = model_files.model.predict_slots_intent(inputs, model_files.tags_vectorizer, 
                                                                               model_files.intents_label_encoder)
    tasks_list = []
    for i in range(len(input_text)):
        for j in range(len(predicted_intents[i])):
            entities = extract_entities(input_text[i], " ".join(predicted_tags[i][j]), jbm_inputs.coordinates[i])
            tasks_list.append(gen_task_card(entities, predicted_intents[i][j]))

    return tasks_list

def predict_tasks_from_obj(data, model_files, bert_vectorizer, logger):

    jbm_inputs = bert_vectorizer.transform(data.text, data.coordinates)

    model = model_files.model

    if (not model.model_params["multi_entity_models"] and model.model_params["use_cls"]) or (model.model_params["multi_entity_models"] and sum(model.model_params["use_cls"]) > 0):
        jbm_inputs.cls_sep_markers = get_cls_sep_markers(data.text, jbm_inputs.valid_positions)

    logger.info(f"Max sequence length: {max(jbm_inputs.sequence_length)}")

    logger.info('.....Predicting....')

    if not model.model_params["use_layout_data"]:
        data_coordinates = []

    if not model.model_params["multi_entity_models"]:
        cards = get_gen_tasks_singleNER(model_files, data.text, jbm_inputs)
    else:
        cards = get_gen_tasks_multiNER(model_files, data.text, jbm_inputs)
    return cards

def predict_task_cards(sess, load_folder_path, data_folder_path, bert_model_hub_path):

    logging.basicConfig(filename="newfile.log", 
                        format='%(asctime)s %(message)s', 
                        filemode='w') 
    logger=logging.getLogger()
    logger.setLevel(logging.INFO) 

    ## loading the model
    logger.info('Loading models ....')
    if not os.path.exists(load_folder_path):
        logger.error('Folder "%s" not exist' % load_folder_path)

    bert_vectorizer = BERTVectorizer(sess, bert_model_hub_path)

    model_files = ModelFiles()

    model_files.model = JointBertModel.load(load_folder_path, sess)

    model_files.intents_label_encoder, _ = load_intents_label_encoder(load_folder_path)
    model_files.tags_vectorizer, _ = load_tags_vectorizer(load_folder_path, model_files.model.model_params["multi_entity_models"], model_files.model.model_params["intents_num"])

    if not model_files.model.model_params["use_layout_data"]:
        data = Reader.read(data_folder_path)
    else:
        data = Reader.read(data_folder_path, get_layout_data=True)

    cards = predict_tasks_from_obj(data, model_files, bert_vectorizer, logger)

    return cards


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
        help='E2E task generation code\n\
        Architecture 1: BERT Base layer with sequence output utilized for NER\n\
        Architecture 2: BERT Sequence output concatenated with Layout embeddings utilized for NER\n\
        Architecture 3: BERT Sequence output concatenated with Layout embeddings processed using Transformer layers before NER inference\n\
        Architecture 4: [CLS] and [SEP] tokens added before and after every line in the token sequence, BERT sequence output for the \
        [CLS] token corresponding to each new line concatenated to the BERT sequence outputs for all tokens in that line before \
        concatenation without layout embeddings and further processing for NER\n\
        Architecture 5: Distinct NER model per intent, specification varies per intent\n')
    parser.add_argument('--model', '-m', help='path to joint bert model', type=str, required=True)
    parser.add_argument('--data', '-d', help='path to test data', type=str, required=True)
    parser.add_argument('--hub_module', '-hm', help='path to tensorflow hub module', type=str, default='https://tfhub.dev/google/small_bert/bert_uncased_L-2_H-128_A-2/1', required=False)
    parser.add_argument('--cache_path', '-cp', help='path to tensorflow hub cache directory', type=str, default="C:\\Users\\ADMIN\\AppData", required=False)

    args = parser.parse_args()
    load_folder_path = args.model
    data_folder_path = args.data
    bert_model_hub_path = args.hub_module
    cache_dir_path = args.cache_path

    if cache_dir_path:
        os.environ['TFHUB_CACHE_DIR'] = cache_dir_path

    sess = tf.compat.v1.Session()

    cards = predict_task_cards(sess, load_folder_path, data_folder_path, bert_model_hub_path)
    print(cards)

    tf.compat.v1.reset_default_graph()