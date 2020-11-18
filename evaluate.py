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
from tags_vectorizer import get_cls_sep_markers, TagsVectorizer, LabelEncoder
from utils import *
from constants import *

# Entity extractor function for evaluation, differs from taskgen extractor since eval requirements are different
def extract_entities(seq_in, ner_output):
    seq_in = seq_in.split()
    ner_output = ner_output.split()
    curr_entity = ner_output[0]
    
    if curr_entity.startswith(BEGIN_ENTITY_PREFIX):
        curr_entity = curr_entity[2:]

    if curr_entity in [BERT_CLS_TOK, BERT_SEP_TOK]:
        curr_entity = OUTSIDE_ENTITY_LABEL
    
    # dictionary of entities containing entity name, its order in the sequence of text, the lines it is spread across, info used for task generation
    extracted_entities = {entity: [] for entity in ENTITY_LIST}
    entity = []
    beg = 0

    for i in range(len(ner_output)):
        if seq_in[i] in [BERT_CLS_TOK, BERT_SEP_TOK]:
            continue
        if ner_output[i] in [BERT_CLS_TOK, BERT_SEP_TOK]:
            ner_output[i] = OUTSIDE_ENTITY_LABEL
        if ner_output[i] == curr_entity:
            if curr_entity != OUTSIDE_ENTITY_LABEL:
                entity.append(seq_in[i])
        else:
            if ner_output[i].startswith(BEGIN_ENTITY_PREFIX) or (not extracted_entities[ner_output[i]]):
                if curr_entity != OUTSIDE_ENTITY_LABEL:
                    extracted_entities[curr_entity].append([entity, beg, i-1])
                entity = [seq_in[i]]
                curr_entity = ner_output[i]
                if ner_output[i].startswith(BEGIN_ENTITY_PREFIX):
                    curr_entity = ner_output[i][2:]
            else:
                extracted_entities[ner_output[i]][-1][0].append(seq_in[i])
                extracted_entities[ner_output[i]][-1][2] = i
    if curr_entity != OUTSIDE_ENTITY_LABEL:
        extracted_entities[curr_entity].append([entity, beg, len(ner_output)])
    for entity in extracted_entities.keys():
        extracted_entities[entity] = [[" ".join(x[0]), x[1], x[2]] for x in extracted_entities[entity]]
    return extracted_entities

def flatten(y):
    ## flatten a list of lists.
    ## flatten([[1,2], [3,4]]) --> [1, 2, 3, 4]
    return list(chain.from_iterable(y))

def get_intents_PR(intents, predicted_intents):
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(predicted_intents)):
        tp += len([x for x in intents[i] if x in predicted_intents[i]])
        fp += len([x for x in predicted_intents[i] if x not in intents[i]])
        fn +=  len([x for x in intents[i] if x not in predicted_intents[i]])
    
    intent_prec = tp/(tp+fp)
    intent_rec = tp/(tp+fn)

    return {'intent_precision': intent_prec, 'intent_recall': intent_rec}

def get_intentwise_PR(intents, predicted_intents):

    intentwise_tp = {}
    intentwise_fp = {}
    intentwise_fn = {}
    intentwise_rec = {}
    intentwise_prec = {}

    for intent in list(set([x for y in intents for x in y])):
        intentwise_tp[intent] = 0
        intentwise_fp[intent] = 0
        intentwise_fn[intent] = 0
        intentwise_rec[intent] = 0
        intentwise_prec[intent] = 0

    for i in range(len(intents)):
        for intent in intents[i]:
            intentwise_tp[intent] += intent in predicted_intents[i]
            intentwise_fn[intent] += intent not in predicted_intents[i]
        for intent in predicted_intents[i]:
            intentwise_fp[intent] += intent not in intents[i]
    
    for intent in list(set([x for y in intents for x in y])):
        if intentwise_tp[intent] + intentwise_fn[intent]:
            intentwise_rec[intent] = intentwise_tp[intent]/(intentwise_tp[intent] + intentwise_fn[intent])
        else:
            intentwise_rec[intent] = None
        if intentwise_tp[intent] + intentwise_fp[intent]:
            intentwise_prec[intent] = intentwise_tp[intent]/(intentwise_tp[intent] + intentwise_fp[intent])
        else:
            intentwise_prec[intent] = None

    return {'intentwise_prec': intentwise_prec, 'intentwise_rec': intentwise_rec}

def get_entitywise_PR(real_tags, pred_tags):

    entitywise_tp = {}
    entitywise_fp = {}
    entitywise_fn = {}
    entitywise_rec = {}
    entitywise_prec = {}
    for entity in list(set(real_tags+pred_tags)):
        entitywise_tp[entity] = 0
        entitywise_fp[entity] = 0
        entitywise_fn[entity] = 0
        entitywise_rec[entity] = 0
        entitywise_prec[entity] = 0
    
    for pred_tag, real_tag in zip(pred_tags, real_tags):
        entitywise_tp[real_tag] += real_tag == pred_tag
        entitywise_fp[pred_tag] += real_tag != pred_tag
        entitywise_fn[real_tag] += real_tag != pred_tag

    for entity in list(set(real_tags)):
        if entitywise_tp[entity] + entitywise_fn[entity]:
            entitywise_rec[entity] = entitywise_tp[entity]/(entitywise_tp[entity] + entitywise_fn[entity])
        else:
            entitywise_rec[entity] = None
        if entitywise_tp[entity] + entitywise_fp[entity]:
            entitywise_prec[entity] = entitywise_tp[entity]/(entitywise_tp[entity] + entitywise_fp[entity])
        else:
            entitywise_prec[entity] = None

    return {'entitywise_precision': entitywise_prec, 'entitywise_recall': entitywise_rec}


def get_entitywise_bleu(input_text, real_tags, pred_tags):

    bleu_score = {}
    for entity in list(set(flatten(real_tags)+flatten(pred_tags))):
        if not entity.startswith(BEGIN_ENTITY_PREFIX):
            bleu_score[entity] = []

    real_entities_list = [extract_entities(x, " ".join(y)) for x, y in zip(input_text, real_tags)]
    pred_entities_list = [extract_entities(x, " ".join(y)) for x, y in zip(input_text, pred_tags)]
    for real_entities, pred_entities in zip(real_entities_list, pred_entities_list):
        pairs = []
        for key in real_entities.keys():
            if key not in pred_entities.keys():
                for entity in real_entities[key]:
                    pairs.append([key, entity[0], ""])
                continue
            for entity in real_entities[key]:
                lap = -1
                ent = ""
                for entity2 in pred_entities[key]:
                    if entity2[1] <= entity[2] and entity2[2] >= entity[1]:
                        if entity2[2] <= entity[2]:
                            if entity2[1] >= entity[1]:
                                if entity2[2] - entity2[1] + 1 > lap:
                                    lap = entity2[2] - entity2[1] + 1
                                    ent = entity2[0]
                            else:
                                if entity2[2] - entity[1] + 1 > lap:
                                    lap = entity2[2] - entity[1] + 1
                                    ent = entity2[0]
                        else:
                            if entity2[1] >= entity[1]:
                                if entity[2] - entity2[1] + 1 > lap:
                                    lap = entity[2] - entity2[1] + 1
                                    ent = entity2[0]
                            else:
                                if entity[2] - entity[1] + 1 > lap:
                                    lap = entity[2] - entity[1] + 1
                                    ent = entity2[0]
                pairs.append([key, entity[0], ent])
            for pair in pairs:
                if pair[2]:
                    score = max(sentence_bleu([[x for x in pair[1].split() if x]], [x for x in pair[2].split() if x], auto_reweigh=True), 
                        sentence_bleu([[x for x in pair[2].split() if x]], [x for x in pair[1].split() if x], auto_reweigh=True))
                    bleu_score[pair[0]].append(score)
                    #if score < 0.5:
                    #   print(pair, score)
                else:
                    bleu_score[pair[0]].append(0)
    for key in bleu_score.keys():
        if bleu_score[key]:
            bleu_score[key] = sum(bleu_score[key])/len(bleu_score[key])

    return {'entitywise_bleu_scores': bleu_score}


## Gives metrics for single NER model architectures
def get_results_singleNER(model_files, jbm_inputs, input_data):
    inputs = [jbm_inputs.input_ids, jbm_inputs.input_mask, jbm_inputs.segment_ids, jbm_inputs.valid_positions]
    if model.use_layout_data:
        inputs.append(jbm_inputs.coordinates)
    if model.use_cls:
        inputs.append(jbm_inputs.cls_sep_markers)
    predicted_tags, predicted_intents = model_files.model.predict_slots_intent(inputs, model_files.tags_vectorizer, model_files.intents_label_encoder, remove_start_end=True)
    
    real_tags = [x.split() for x in input_data.tags]
    real_tags = [real_tags[i][:len(predicted_tags[i])] for i in range(len(real_tags))]

    metrics = {}
    
    metrics["entity_accuracy"] = metrics.accuracy_score(flatten(real_tags), flatten(predicted_tags))

    metrics.update(get_entitywise_PR(flatten(real_tags), flatten(predicted_tags)))
	
    metrics.update(get_entitywise_bleu(input_data.text, real_tags, predicted_tags))

## Gives metrics for architectures having intent specific separate NER model 
def get_results_multiNER(model_files, jbm_inputs, input_data):
    inputs = [jbm_inputs.input_ids, jbm_inputs.input_mask, jbm_inputs.segment_ids, jbm_inputs.valid_positions, jbm_inputs.coordinates]
    if sum(model_files.model.model_params["use_cls"]) > 0:
        inputs.append(jbm_inputs.cls_sep_markers)
    predicted_tags, predicted_intents = model_files.model.predict_slots_intent(inputs, model_files.tags_vectorizer, 
                                                                               model_files.intents_label_encoder)
    #print(predicted_tags, '\n', predicted_intents)
    metrics = {}

    metrics.update(get_intents_PR(input_data.intents, predicted_intents))

    pred_tags = []
    real_tags = []
    tags_nf = []
    pred_tags_sep = []
    real_tags_sep = []
    input_text_sep = []
    for i in range(len(input_data.intents)):
        for j in range(len(input_data.intents[i])):
            if input_data.intents[i][j] in predicted_intents[i]:
                pred_tags.extend(predicted_tags[i][predicted_intents[i].index(input_data.intents[i][j])])
                real_tags.extend(input_data.tags[i][j].split()[:len(predicted_tags[i][predicted_intents[i].index(input_data.intents[i][j])])])
                pred_tags_sep.append(predicted_tags[i][predicted_intents[i].index(input_data.intents[i][j])])
                real_tags_sep.append(input_data.tags[i][j].split()[:len(pred_tags_sep[-1])])
                input_text_sep.append(input_data.text[i])
            else:
                tags_nf.append(input_data.tags[i][j].split())

    metrics.update(get_intentwise_PR(input_data.intents, predicted_intents))

    metrics.update({"entity_accuracy": sum([x==y for x, y in zip(pred_tags, real_tags)])/len(real_tags)})

    metrics.update(get_entitywise_PR(real_tags, pred_tags))

    metrics.update(get_entitywise_bleu(input_text_sep, real_tags_sep, pred_tags_sep))

    return metrics

def evaluate(sess, load_folder_path, data_folder_path, bert_model_hub_path):

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

    model = model_files.model

    model_files.intents_label_encoder, _ = load_intents_label_encoder(load_folder_path)
    model_files.tags_vectorizer, _ = load_tags_vectorizer(load_folder_path, model.model_params["multi_entity_models"], model.model_params["intents_num"])

    if not model.model_params["use_layout_data"]:
        data = Reader.read(data_folder_path, get_layout_data=False)
    else:
        data = Reader.read(data_folder_path, get_layout_data=True)    
    
    jbm_inputs = bert_vectorizer.transform(data.text, data.coordinates)

    if (not model.model_params["multi_entity_models"] and model.model_params["use_cls"]) or (model.model_params["multi_entity_models"] and sum(model.model_params["use_cls"]) > 0):
        jbm_inputs.cls_sep_markers = get_cls_sep_markers(data.text, jbm_inputs.valid_positions)

    logger.info(f"Longest sequence length: {max(jbm_inputs.sequence_length)}")

    logger.info('.....Evaluation....')

    if not model.model_params["multi_entity_models"]:
        return get_results_singleNER(model_files, jbm_inputs, data)
    
    data.tags = [[x] for x in data.tags]
    data.intents = [[x] for x in data.intents]
    return get_results_multiNER(model_files, jbm_inputs, data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
        help='Evaluation code to gove intentwise P/R metrics and entitywise P, R, modified BLEU metrics.\n\
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

    metrics = evaluate(sess, load_folder_path, data_folder_path, bert_model_hub_path)
    print(json.dumps(metrics))

    tf.compat.v1.reset_default_graph()