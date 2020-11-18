from Reader_data import Reader
from bert_vectorizer import BERTVectorizer
from tags_vectorizer import TagsVectorizer, get_cls_sep_markers, LabelEncoder
from joint_bert_model import JointBertModel
import numpy as np
import argparse
import tensorflow as tf
import os
import json
import logging

from utils import *
from constants import intentwise_architecture, architecture_specs
# from keras import backend as K

def get_train_val_data(sess, train_data_folder_path, val_data_folder_path, bert_model_hub_path, architecture):

    ## import the dataset
    if architecture != 1:
        train_data = Reader.read(train_data_folder_path, get_layout_data=True) ## train dataset
        val_data = Reader.read(val_data_folder_path, get_layout_data=True) ## validation dataset
    else:
        train_data = Reader.read(train_data_folder_path, get_layout_data=False) ## train dataset
        val_data = Reader.read(val_data_folder_path, get_layout_data=False) ## validation dataset

    ## vectorize the train_text_arr and val_text_arr
    bert_vectorizer = BERTVectorizer(sess, bert_model_hub_path)

    train_jbm_inputs = bert_vectorizer.transform(train_data.text, train_data.coordinates)
    val_jbm_inputs = bert_vectorizer.transform(val_data.text, val_data.coordinates)

    ## encode the intents label, directly by using the LabelEncoder library, which is provided by skeleran
    intents_label_encoder = LabelEncoder()
    intents_label_encoder.fit(train_data.intents + val_data.intents)
    train_data.intents = intents_label_encoder.transform(train_data.intents).astype(np.int32)

    # ## we should use the train dataset to fit the label encoder and then return the encoded labels
    # train_intents = intents_label_encoder.fit_transform(train_intents).astype(np.int32) ## fit_transform

    val_data.intents = intents_label_encoder.transform(val_data.intents).astype(np.int32) ## transform
    intents_num = len(intents_label_encoder.tags)

    ## cls sep marker information required only if any of the intent's NER
    ## model is based on architecture 4 in case of multi-intent NER models
    if architecture == 4 or (architecture == 5 and 4 in intentwise_architecture.values()):
        train_jbm_inputs.cls_sep_markers = get_cls_sep_markers(train_data.tags, train_jbm_inputs.valid_positions)
        val_jbm_inputs.cls_sep_markers = get_cls_sep_markers(val_data.tags, val_jbm_inputs.valid_positions)

    return {'train_data': train_data, 'val_data': val_data, 'train_jbm_inputs': train_jbm_inputs, 'val_jbm_inputs': val_jbm_inputs}, intents_label_encoder, intents_num

def tags_vectorization(data, intents_num, architecture):
    if architecture in [1, 2, 3, 4]:
        tags_vectorizer = TagsVectorizer()
        tags_vectorizer.fit(data["train_data"].tags, data["val_data"].tags) ## use the train dataset to fit the tagsvectorizer
        data["train_data"].tags = tags_vectorizer.transform(data["train_data"].tags, data["train_data"].valid_positions)
        data["val_data"] = tags_vectorizer.transform(data["val_data"].tags, data["val_data"].valid_positions)
        slots_num = len(tags_vectorizer.label_encoder.tags)
    else:
        tags_vectorizers = []
        train_tags = []
        val_tags = []

        for _ in range(intents_num):
            train_tags.append([])
            for i in range(len(data["train_data"].tags)):
                train_tags[-1].append([-1] * len(data["train_jbm_inputs"].valid_positions[i]))

        for _ in range(intents_num):
            val_tags.append([])
            for i in range(len(data["val_data"].tags)):
                val_tags[-1].append([-1] * len(data["val_jbm_inputs"].valid_positions[i]))

        slots_num = []
        for i in range(intents_num):
            tags_vectorizer = TagsVectorizer()
            train_idxs = [j for j in range(len(data["train_data"].intents)) if data["train_data"].intents[j] == i]
            train_tags_i = [data["train_data"].tags[j] for j in train_idxs]
            train_valid_positions_i = [data["train_jbm_inputs"].valid_positions[j] for j in train_idxs]
            val_idxs = [j for j in range(len(data["val_data"].intents)) if data["val_data"].intents[j] == i]
            val_tags_i = [data["val_data"].tags[j] for j in val_idxs]
            val_valid_positions_i = [data["val_jbm_inputs"].valid_positions[j] for j in val_idxs]
            tags_vectorizer.fit(train_tags_i, val_tags_i)
            train_tags_i = tags_vectorizer.transform(train_tags_i, train_valid_positions_i)
            for j in range(len(train_idxs)):
                train_tags[i][train_idxs[j]] = train_tags_i[j]
            val_tags_i = tags_vectorizer.transform(val_tags_i, val_valid_positions_i)
            for j in range(len(val_idxs)):
                val_tags[i][val_idxs[j]] = val_tags_i[j]
            slots_num.append(len(tags_vectorizer.label_encoder.tags))
            tags_vectorizers.append(tags_vectorizer)

        for i in range(intents_num):
            for j in range(len(data["train_data"].tags)):
                train_tags[i][j] = np.array(train_tags[i][j])
            train_tags[i] = np.array(train_tags[i])
            for j in range(len(data["val_data"].tags)):
                val_tags[i][j] = np.array(val_tags[i][j])
            val_tags[i] = np.array(val_tags[i])

        data["train_data"].tags = train_tags
        data["val_data"].tags = val_tags
        tags_vectorizer = tags_vectorizers

        return tags_vectorizer, slots_num

def get_model(sess, intents_num, slots_num, architecture, intents_label_encoder):
    if architecture in [1, 2, 3, 4]:
        specs = architecture_specs[architecture]
        model_params = {
            'slots_num': slots_num,
            'intents_num': intents_num,
            'num_bert_fine_tune_layers': 12,
            'entities_dnn_depth': specs['entities_dnn_depth'],
            'transformer_block_depth': specs['transformer_block_depth'],
            'intents_dnn_depth': specs['intents_dnn_depth'],
            'use_layout_data': specs['use_layout_data'],
            'multi_entity_models': False,
            'use_cls': specs['use_cls']
        }
        model = JointBertModel(sess, model_params)
    if architecture == 5:
        ## all NER model attributes can vary for each intent
        ## hence all attributes are modelled as lists
        ## ith item in an attribute list will correspond
        ## to ith intent in the intent label encoder's tags list
        model_params = {
            'slots_num': slots_num,
            'intents_num': intents_num,
            'num_bert_fine_tune_layers': 12,
            'entities_dnn_depth': [],
            'transformer_block_depth': [],
            'intents_dnn_depth': 1,
            'use_layout_data': [],
            'multi_entity_models': True,
            'use_cls': []
        }
        for i in range(intents_num):
            ## By default, architecture 3 is assigned because
            ## it is the best performing general architecture
            if intents_label_encoder.tags[i] in intentwise_architecture.keys():
                spec = architecture_specs[intentwise_architecture[intents_label_encoder.tags[i]]]
            else:
                spec = architecture_specs[3]
            model_params['entities_dnn_depth'].append(spec['entities_dnn_depth'])
            model_params['transformer_block_depth'].append(spec['transformer_block_depth'])
            model_params['use_layout_data'].append(spec['use_layout_data'])
            model_params['use_cls'].append(spec['use_cls'])
        model = JointBertModel(sess, model_params)

    return model


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

def train(sess, train_data_folder_path, val_data_folder_path, save_folder_path, epochs, batch_size, bert_model_hub_path, architecture):

    logging.basicConfig(filename="newfile.log", 
                        format='%(asctime)s %(message)s', 
                        filemode='w') 
    logger=logging.getLogger()
    logger.setLevel(logging.INFO)

    ## getting train and val data
    data, intents_label_encoder, intents_num = get_train_val_data(sess, train_data_folder_path, val_data_folder_path, bert_model_hub_path, architecture)

    ## vectorize the train data tags and val data tags
    tags_vectorizer, slots_num = tags_vectorization(data, intents_num, architecture)

    logger.info(f"Number of slots: {slots_num}")
    logger.info(f"Number of intents: {intents_num}")

    ## getting the joint bert model
    model = get_model(sess, intents_num, slots_num, architecture, intents_label_encoder)

    model_data = {}
    model_data["input"] = {"train": [data["train_jbm_inputs"].input_ids, data["train_jbm_inputs"].input_mask, data["train_jbm_inputs"].segment_ids, data["train_jbm_inputs"].valid_positions],
                           "val": [data["val_jbm_inputs"].input_ids, data["val_jbm_inputs"].input_mask, data["val_jbm_inputs"].segment_ids, data["val_jbm_inputs"].valid_positions]}
    model_data["output"] = {"train": [data["train_data"].tags, data["train_data"].intents], "val": [data["val_data"].tags, data["val_data"].intents]}


    if architecture in [2, 3]:
        model_data["input"]["train"].append(data["train_jbm_inputs"].coordinates)
        model_data["input"]["val"].append(data["val_jbm_inputs"].coordinates)
    if architecture == 4:
        model_data["input"]["train"].extend([data["train_jbm_inputs"].coordinates, data["train_jbm_inputs"].cls_sep_markers])
        model_data["input"]["val"].extend([data["val_jbm_inputs"].coordinates, data["val_jbm_inputs"].cls_sep_markers])
    if architecture == 5:
        model_data["input"]["train"].append(data["train_jbm_inputs"].coordinates)
        model_data["input"]["val"].append(data["val_jbm_inputs"].coordinates)
        model_data["input"]["train"] += (4 in intentwise_architecture.values()) * [data["train_jbm_inputs"].cls_sep_markers]
        model_data["input"]["val"] += (4 in intentwise_architecture.values()) * [data["val_jbm_inputs"].cls_sep_markers]
        model_data["output"]["train"] = [data["train_data"].intents] + data["train_data"].tags
        model_data["output"]["val"] = [data["val_data"].intents] + data["val_data"].tags

    model.fit(model_data["input"]["train"], model_data["output"]["train"], validation_data=(model_data["input"]["val"], model_data["output"]["val"]),
              epochs=epochs, batch_size=batch_size)

    ## saving
    logger.info('saving...')
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
        logger.info('Folder "%s" created' % save_folder_path)
    model.save(save_folder_path)

    frozen_graph = freeze_session(sess, output_names=[out.op.name for out in model.model.outputs])

    tf.train.write_graph(frozen_graph, save_folder_path, "saved_model.pb", as_text=False)

    if architecture in [1, 2, 3, 4]:
        with open(os.path.join(save_folder_path, 'tags_vectorizer.json'), 'w') as f:
            json.dump(tags_vectorizer.label_encoder.tags, f)
    else:
        for i, vectorizer in enumerate(tags_vectorizer):
            with open(os.path.join(save_folder_path, 'tags_vectorizer_{i}.json'), 'w') as f:
                json.dump(vectorizer.label_encoder.tags, f)

    with open(os.path.join(save_folder_path, 'intents_label_encoder.json'), 'w') as f:
        json.dump(intents_label_encoder.tags, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training the Joint Slot filling and Intent classification based on Bert', add_help=False)
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
        help='Training the Joint Slot filling and Intent classification based on Bert\n\
        Architecture 1: BERT Base layer with sequence output utilized for NER\n\
        Architecture 2: BERT Sequence output concatenated with Layout embeddings utilized for NER\n\
        Architecture 3: BERT Sequence output concatenated with Layout embeddings processed using Transformer layers before NER inference\n\
        Architecture 4: [CLS] and [SEP] tokens added before and after every line in the token sequence, BERT sequence output for the \
        [CLS] token corresponding to each new line concatenated to the BERT sequence outputs for all tokens in that line before \
        concatenation without layout embeddings and further processing for NER\n\
        Architecture 5: Distinct NER model per intent, specification varies per intent\n')
    parser.add_argument('--train', '-t', help='path to training data', type=str, required=True)
    parser.add_argument('--val', '-v', help='path to validation data', type=str, required=True)
    parser.add_argument('--save', '-s', help='folder path to save the trained model', type=str, required=True)
    parser.add_argument('--epochs', '-e', help='number of epochs', type=int, default=5, required=False)
    parser.add_argument('--batch_size', '-bs', help='batch size', type=int, default=64, required=False)
    parser.add_argument('--hub_module', '-hm', help='path to tensorflow hub module', type=str, default='https://tfhub.dev/google/small_bert/bert_uncased_L-2_H-128_A-2/1', required=False)
    parser.add_argument('--cache_path', '-cp', help='path to tensorflow hub cache directory', type=str, default="C:\\Users\\ADMIN\\AppData", required=False)
    parser.add_argument('--architecture', '-a', help='architecture of the model', type=int, default=5, required=False)

    args = parser.parse_args()
    train_data_folder_path = args.train
    val_data_folder_path = args.val
    save_folder_path = args.save
    epochs = args.epochs
    batch_size = args.batch_size
    bert_model_hub_path = args.hub_module
    cache_dir_path = args.cache_path
    architecture = args.architecture

    if cache_dir_path:
        os.environ['TFHUB_CACHE_DIR'] = cache_dir_path

    tf.compat.v1.random.set_random_seed(7)
    sess = tf.compat.v1.Session()

    train(sess, train_data_folder_path, val_data_folder_path, save_folder_path, epochs, batch_size, bert_model_hub_path, architecture)

    tf.compat.v1.reset_default_graph()