import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Multiply, TimeDistributed, Dropout, Concatenate, RNN
from tensorflow.python.keras.metrics import Metric
from bert_model import BertLayer
from nlu_model import NLUModel
import numpy as np
import os
import json
import tensorflow.keras.backend as K
from tensorflow.python.ops.array_ops import gather
from layers import Transformer, CLSProcess

'''
build and compile our model using the BERT layer
'''


def acc(y_true, y_pred):
    return tf.divide(tf.cast(tf.reduce_sum(tf.multiply(tf.cast(tf.reshape(K.greater(y_true, -1), [-1]), dtype=tf.int64), 
    	tf.cast(K.equal(tf.reshape(tf.cast(y_true, dtype=tf.int64), [-1]), tf.reshape(K.argmax(y_pred, axis=-1), [-1])), 
    	dtype=tf.int64))), tf.float64), tf.cast(tf.reduce_sum(tf.reshape(tf.cast(K.greater(y_true, -1), dtype=tf.int64), [-1])), tf.float64) + K.epsilon())

def get_sparse_categorical_focal_loss(depth):

    def sparse_categorical_focal_loss(y_true, y_pred):
        #print(y_true)
        valid = tf.reshape(tf.cast(K.greater(y_true, -1), dtype=tf.float32), [-1])
        y_true = tf.one_hot(tf.cast(tf.reshape(y_true,[-1]), dtype=tf.int32), depth=depth, axis=-1)
        y_pred = tf.stack([tf.reshape(x, [-1]) for x in tf.unstack(y_pred, axis=-1)], axis=-1)
        probs = tf.reduce_sum(tf.math.multiply(y_pred, y_true), axis=-1)
        #print(probs, y_true, y_pred)
        loss = -tf.reduce_sum(tf.reshape((1 - probs) ** 1.5 * tf.log(probs+K.epsilon()) * valid, [-1])) / (tf.reduce_sum(tf.reshape(valid, [-1])) + K.epsilon())
        #print(loss)
        return loss

    return sparse_categorical_focal_loss

class JointBertModel(NLUModel):
    def __init__(self, sess, params):
        self.model_params = params
        
        if params['multi_entity_models']:
            self.model_params['entities_dnn_depth'] = self.paramToList(params['entities_dnn_depth'])
            self.model_params['transformer_block_depth'] = self.paramToList(params['transformer_block_depth'])
            self.model_params['use_layout_data'] = self.paramToList(params['use_layout_data'])
            self.model_params['use_cls'] = self.paramToList(params['use_cls'])

        self.build_model()
        self.compile_model()

        self.initialize_vars(sess)


    def get_single_NER_output(self, bert_sequence_output):

        slots_fc = bert_sequence_output
        # concatenate the cls token output to the associated line if cls vector is to be used
        if self.model_params['use_cls']:
            concat_layer = RNN(CLSProcess(128), return_sequences=True)
            concat = concat_layer(Concatenate(axis=-1)([self.model_inputs['in_cls_sep'], slots_fc]))
            slots_fc = Concatenate(axis=-1)([slots_fc, concat])
            slots_fc = TimeDistributed(Dense(128))(slots_fc)

        transformer_hidden_size = 128
        if self.model_params['use_layout_data']:
            coord_fc = TimeDistributed(Dense(128, activation='relu'))(self.model_inputs['in_tok_coordinates'])
            slots_fc = Concatenate(axis=-1)([slots_fc, coord_fc])
            transformer_hidden_size = 256

        for _ in range(self.model_params['transformer_block_depth']):
            layer = Transformer(num_hidden_layers=1,
                                hidden_size=transformer_hidden_size,
                                num_attention_heads=2,
                                intermediate_size=transformer_hidden_size,
                                intermediate_activation="relu")
            slots_fc = layer(slots_fc)
        for _ in range(self.model_params['entities_dnn_depth'] - 1):
            slots_fc = TimeDistributed(Dense(512, activation='relu'))(slots_fc)
        slots_output = TimeDistributed(Dense(self.model_params['slots_num'], activation='softmax'))(slots_fc)
        slots_output = Multiply(name='slots_tagger')([slots_output, self.model_inputs['in_valid_positions']])

        return Model(inputs=model_inputs.values(), outputs=[slots_output, intents_fc])

    def get_multi_NER_output(self, bert_sequence_output):

        for i in range(self.model_params['intents_num']):
            # concatenate the cls token output to the associated line if cls vector is to be used
            slots_fc = bert_sequence_output
            if self.model_params['use_cls'][i]:
                concat_layer = RNN(CLSProcess(128), return_sequences=True)
                concat = concat_layer(Concatenate(axis=-1)([self.model_inputs['in_cls_sep'], slots_fc]))
                slots_fc = Concatenate(axis=-1)([slots_fc, concat])
                slots_fc = TimeDistributed(Dense(128))(slots_fc)

            transformer_hidden_size = 128
            if self.model_params['use_layout_data'][i]:
                coord_fc = TimeDistributed(Dense(128, activation='relu'))(self.model_inputs['in_tok_coordinates'])
                slots_fc = Concatenate(axis=-1)([slots_fc, coord_fc])
                transformer_hidden_size = 256

            for _ in range(self.model_params['transformer_block_depth'][i]):
                layer = Transformer(num_hidden_layers=1,
                    hidden_size=transformer_hidden_size,
                    num_attention_heads=2,
                    intermediate_size=256,
                    intermediate_activation="relu")
                slots_fc = layer(slots_fc)
            
            for _ in range(self.model_params['entities_dnn_depth'][i] - 1):
                slots_fc = TimeDistributed(Dense(512, activation='relu'))(slots_fc)
            slots_output = TimeDistributed(Dense(self.model_params['slots_num'][i], activation='softmax'))(slots_fc)
            valid_positions = tf.expand_dims(self.model_inputs['in_valid_positions'], axis=2) ## expand the shape of the array to axis=2
            ## 3-D in_valid_position
            valid_positions = Concatenate()([valid_positions] * self.model_params['slots_num'][i])
            slots_output = Multiply(name='slots_tagger_'+str(i+1))([slots_output, valid_positions])
            self.model_outputs.append(slots_output)

    def build_model(self):

        self.model_inputs = {}
        self.model_inputs['in_id'] = Input(shape=(None,), name='input_ids')
        self.model_inputs['in_mask'] = Input(shape=(None,), name='input_masks')
        self.model_inputs['in_segment'] = Input(shape=(None,), name='segment_ids')
        if self.model_params['multi_entity_models']:
            self.model_inputs['in_valid_positions'] = Input(shape=(None,), name='valid_positions')
        else:
            self.model_inputs['in_valid_positions'] = Input(shape=(None, self.model_params['slots_num']), name='valid_positions')
        if self.model_params['use_layout_data']:
            self.model_inputs['in_tok_coordinates'] = Input(shape=(None, 4), name='token_coordinates')
        if self.model_params['use_cls']:
            self.model_inputs['in_cls_sep'] = Input(shape=(None, 2), name='cls_sep_tok_markers')
        bert_inputs = [self.model_inputs[x] for x in ['in_id', 'in_mask', 'in_segment', 'in_valid_positions']]

        # the output of trained Bert
        bert_pooled_output, bert_sequence_output = BertLayer(n_fine_tune_layers=self.model_params['num_bert_fine_tune_layers'], name='BertLayer')(bert_inputs)

        # add the additional layer for intent classification and slot filling
        # intents_drop = Dropout(rate=0.1)(bert_pooled_output)
        intents_fc = bert_pooled_output
        for _ in range(self.model_params['intents_dnn_depth'] - 1):
            intents_fc = Dense(128, activation='relu')(intents_fc)
        intents_fc = Dense(self.model_params['intents_num'], activation='sigmoid', name='intent_classifier')(intents_fc)

        #slots_drop = Dropout(rate=0.1)(bert_sequence_output)

        if not self.model_params['multi_entity_models']:
            slots_output = self.get_single_NER_output(bert_sequence_output)
            self.model = Model(inputs=model_inputs.values(), outputs=[slots_output, intents_fc])

        else:
            self.model_outputs = [intents_fc]
            self.get_multi_NER_output(bert_sequence_output)
            self.model = Model(inputs=self.model_inputs.values(), outputs=self.model_outputs)
            

    def compile_model(self):

        optimizer = tf.keras.optimizers.Adam(lr=5e-5)
        # if the targets are one-hot labels, using 'categorical_crossentropy'; while if targets are integers, using 'sparse_categorical_crossentropy'
        if not self.model_params['multi_entity_models']:
            losses = {
                'slots_tagger': get_sparse_categorical_focal_loss(self.model_params['slots_num']),
                'intent_classifier': 'sparse_categorical_crossentropy'
            }
            ## loss_weights: to weight the loss contributions of different model outputs.
            loss_weights = {'slots_tagger': 4.0, 'intent_classifier': 0.0}
            # slot_tagger_acc = SlotTaggerAccuracy()
            metrics = {'slots_tagger': acc, 'intent_classifier': 'acc'}
        else:
            losses = {
                'intent_classifier': 'sparse_categorical_crossentropy'
            }
            for i in range(self.model_params['intents_num']):
                losses["slots_tagger_"+str(i+1)] = get_sparse_categorical_focal_loss(depth=self.model_params['slots_num'][i])
            
            ## loss_weights: to weight the loss contributions of different model outputs.
            loss_weights = {'intent_classifier': 1.0}
            for i in range(self.model_params['intents_num']):
                loss_weights["slots_tagger_"+str(i+1)] = 2.0

            metrics = {'intent_classifier': 'acc'}
            for i in range(self.model_params['intents_num']):
                metrics["slots_tagger_"+str(i+1)] = acc

        self.model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)

        self.model.summary()


    def fit(self, X, Y, validation_data=None, epochs=5, batch_size=32):
        if not self.model_params['multi_entity_models']:
            X[3] = self.prepare_valid_positions(X[3])
            validation_data[0][3] = self.prepare_valid_positions(validation_data[0][3])

        history = self.model.fit(X, Y, validation_data=validation_data, epochs=epochs, batch_size=batch_size)

        # self.visualize_metric(history.history, 'slots_tagger_loss')
        # self.visualize_metric(history.history, 'intent_classifier_loss')
        # self.visualize_metric(history.history, 'loss')
        # self.visualize_metric(history.history, 'intent_classifier_acc')


    def prepare_valid_positions(self, in_valid_positions):
        ## the input is 2-D in_valid_position
        in_valid_positions = np.expand_dims(in_valid_positions, axis=2) ## expand the shape of the array to axis=2
        ## 3-D in_valid_position
        in_valid_positions = np.tile(in_valid_positions, (1,1,self.model_params['slots_num'])) ##
        return in_valid_positions

    def predict_slots_intent(self, x, slots_vectorizer, intent_vectorizer, remove_start_end=True, remove_cls_sep=True):
        if not self.model_params['multi_entity_models']:
            valid_positions = x[3]
            x[3] = self.prepare_valid_positions(x[3])

            y_slots, y_intent = self.predict(x)

            ### get the real slot-tags using 'inverse_transform' of slots-vectorizer
            slots = slots_vectorizer.inverse_transform(y_slots, valid_positions)
            if remove_start_end: ## remove the first '[CLS]' and the last '[SEP]' tokens.
                slots = np.array([x[1:-1] for x in slots])
            if remove_cls_sep:
                slots = np.array([[y for y in x if y not in ['[CLS]', '[SEP]']] for x in slots])

            ### get the real intents using 'inverse-transform' of intents-vectorizer
            intents = np.array([intent_vectorizer.inverse_transform([np.argmax(y_intent[i])])[0] for i in range(y_intent.shape[0])])
        
        else:
            valid_positions = x[3]
            output = self.predict(x)
            y_intents = output[0]
            y_slots = []
            for i in range(self.model_params['intents_num']):
                y_slots.append(output[i+1])
            ### get the real slot-tags using 'inverse_transform' of slots-vectorizer
            slots_all = []
            for i in range(self.model_params['intents_num']):
                slots_i = slots_vectorizer[i].inverse_transform(y_slots[i], valid_positions)
                if remove_start_end: ## remove the first '[CLS]' and the last '[SEP]' tokens.
                    slots_i = [x[1:-1] for x in slots_i]
                slots_all.append(slots_i)

            slots = []
            for _ in range(y_intents.shape[0]):
                slots.append([])
            for i in range(self.model_params['intents_num']):
                for j in range(y_intents.shape[0]):
                    if y_intents[j][i] >= 0.5:
                        slots[j].append(slots_all[i][j])

            ### get the real intents using 'inverse-transform' of intents-vectorizer
            intents = [[intent_vectorizer.inverse_transform([j])[0] for j in range(self.model_params['intents_num']) if y_intents[i][j] >= 0.5] for i in range(y_intents.shape[0])]

        return slots, intents

    def initialize_vars(self, sess):
        sess.run(tf.compat.v1.local_variables_initializer())
        sess.run(tf.compat.v1.global_variables_initializer())
        K.set_session(sess)

    def save(self, model_path):
        with open(os.path.join(model_path, 'params.json'), 'w') as json_file:
            json.dump(self.model_params, json_file)
        self.model.save_weights(os.path.join(model_path, 'joint_bert_model.h5'))

    def load(load_folder_path, sess):
        with open(os.path.join(load_folder_path, 'params.json'), 'r') as json_file:
            model_params = json.load(json_file)

        new_model = JointBertModel(sess, model_params)
        new_model.model.load_weights(os.path.join(load_folder_path, 'joint_bert_model.h5'))
        return new_model

    def paramToList(self, param):
        if isinstance(param, list):
            return param
        else:
            return [param] * self.intents_num

    def get_config(self):

        config = super().get_config().copy()
        config.update({
                'model_params': self.model_params
            })
        return config