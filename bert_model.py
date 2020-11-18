import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.python.keras.backend as K

'''
custom BERT Layer in Keras, pre trained BERT weights imported from tf-hub
'''
class BertLayer(tf.keras.layers.Layer):
    def __init__(self, n_fine_tune_layers=12, bert_path='https://tfhub.dev/google/small_bert/bert_uncased_L-2_H-128_A-2/1', **kwargs):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.bert_path = bert_path
        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(self.bert_path, trainable=True, name="{}_module".format(self.name))
        trainable_vars = self.bert.variables
        # Remove unused layers
        trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
        # Select how many layers to fine tune
        #trainable_vars = trainable_vars[-self.n_fine_tune_layers:]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)
        # Add non_trainable weights:
        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs] #cast the variables to int32 tensor
        input_ids, input_mask, segment_ids, valid_positions = inputs
        ## we don't feed the valid_position into the model, the valid_position is only used for the slots transform to align to the tokenized input
        bert_inputs = dict(input_ids=input_ids, ## we can use 'convert_tokens_to_ids' function to get the ids from tokens
                           input_mask=input_mask,
                           segment_ids=segment_ids)
        result = self.bert(inputs=bert_inputs, signature='tokens', as_dict=True)
        return result['pooled_output'], result['sequence_output']

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 128)
        
    def get_config(self):
    
        config = super().get_config().copy()
        config.update({
            'n_fine_tune_layers': self.n_fine_tune_layers,
            'bert_path': self.bert_path
        })
        return config