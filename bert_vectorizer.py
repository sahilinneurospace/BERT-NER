import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from bert.tokenization import FullTokenizer
import typing
from typing import List, Tuple
from utils import *

'''
import the BERT model from tensorflow_hub library and create the tokenizer to pre-process the dataset,
and to tokenize every word and get the corresponding input_ids, input_mask, segment_ids and valid_positions of every sentence
'''
class BERTVectorizer:

    def __init__(self, sess, bert_model_hub_path):
        self.sess = sess
        self.bert_model_hub_path = bert_model_hub_path
        self.create_tokenizer_from_hub_module()

    def create_tokenizer_from_hub_module(self):
        # get the vocabulary and lowercasing or uppercase information directly from the BERT tf hub module
        bert_module = hub.Module(self.bert_model_hub_path)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        vocab_file, do_lower_case = self.sess.run(
            [
                tokenization_info["vocab_file"],
                tokenization_info["do_lower_case"]
            ]
        )
        self.tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case) #do_lower_case=True
        # print(tokenizer.tokenize('hello world!'))  --> ['hello', 'world', '!']

    def tokenize(self, text:str, coords): ## tokenize every sentence
        words = text.split()
        ## # text: add leah kauffman to my uncharted 4 nathan drake playlist
        ## # words: ['add', 'leah', 'kauffman', 'to', 'my', 'uncharted', '4', 'nathan', 'drake', 'playlist']
        tokens = []
        ## # tokens: ['add', 'leah', 'ka', '##uf', '##fm', '##an', 'to', 'my', 'un', '##cha', '##rted', '4', 'nathan', 'drake', 'play', '##list']
        valid_positions = []
        if coords:
            coordinates = []
            ## # valid_positions:[1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0]
            for word, coord in zip(words, coords):
                token = self.tokenizer.tokenize(word)
                dist = [len(t)/len(word) for t in token]
                dist = [dist[0]] + [x - 2/len(word) for x in dist[1:]]
                x0, y0, x1, y1 = coord
                for r in dist:
                    coordinates.append([x0, y0, x0 + r * (x1 - x0), y1])
                    x0 = x0 + r * (x1 - x0)
                tokens.extend(token)
                for i in range(len(token)):
                    if i == 0:
                        valid_positions.append(1)
                    else:
                        valid_positions.append(0)
            return tokens, valid_positions, coordinates

        for word in words:
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for i in range(len(token)):
                if i == 0:
                    valid_positions.append(1)
                else:
                    valid_positions.append(0)
        return tokens, valid_positions, []

    def __vectorize(self, text:str, coords=[]):
        tokens, valid_positions, coords = self.tokenize(text, coords)

        ## insert the first token "[CLS]"
        tokens.insert(0, '[CLS]')
        valid_positions.insert(0, 1)
        ## insert the last token "[SEP]"
        tokens.append('[SEP]')
        valid_positions.append(1)

        if coords:
            coords.insert(0, [0.] * 4)
            coords.append([0.] * 4)
        ## ['[CLS]', 'add', 'leah', 'ka', '##uf', '##fm', '##an', 'to', 'my', 'un', '##cha', '##rted', '4', 'nathan', 'drake', 'play', '##list', '[SEP]']
        ## [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1]

        '''
        (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0

        Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        '''
        segment_ids = [0] * len(tokens)
        ## # segment_ids: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ## # input_ids: [101, 5587, 14188, 10556, 16093, 16715, 2319, 2000, 2026, 4895, 7507, 17724, 1018, 7150, 7867, 2377, 9863, 102] and the first is always 101 and the last is 102

        input_mask = [1] * len(input_ids) ## The mask has 1 for real tokens and 0 for padding tokens.
        ## # input_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        return input_ids, input_mask, segment_ids, valid_positions, coords

    def transform(self, text_arr, coords_list=[]):
        input_ids = []
        input_mask = []
        segment_ids = []
        valid_positions = []
        if coords_list:
            input_coordinates = []
            for text, coords in zip(text_arr, coords_list):
                ids, mask, seg_ids, valid_pos, coordinates = self.__vectorize(text, coords)
                input_ids.append(ids)
                input_mask.append(mask)
                segment_ids.append(seg_ids)
                valid_positions.append(valid_pos)
                input_coordinates.append(coordinates)
        else:
            for text in text_arr:
                ids, mask, seg_ids, valid_pos, _ = self.__vectorize(text)
                input_ids.append(ids)
                input_mask.append(mask)
                segment_ids.append(seg_ids)
                valid_positions.append(valid_pos)

        sequence_length = np.array([len(i) for i in input_ids])

        ## set the maximum length to 512
        input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids, maxlen=512, truncating='post', padding='post')
        input_mask = tf.keras.preprocessing.sequence.pad_sequences(input_mask, maxlen=512, truncating='post', padding='post')
        segment_ids = tf.keras.preprocessing.sequence.pad_sequences(segment_ids, maxlen=512, truncating='post', padding='post')
        valid_positions = tf.keras.preprocessing.sequence.pad_sequences(valid_positions, maxlen=512, truncating='post', padding='post')

        bert_inputs = BertInputs(input_ids, input_mask, segment_ids, valid_positions)
        jbm_inputs = JointBertModelInputs(bert_inputs=bert_inputs, sequence_length=sequence_length)

        if coords_list:
            for i in range(len(input_coordinates)):
                if len(input_coordinates[i]) > 512:
                    input_coordinates[i] = input_coordinates[i][:512]
                else:
                    input_coordinates[i].extend([[0.] * 4] * (512 - len(input_coordinates[i])))
            input_coordinates = np.array(input_coordinates)
            jbm_inputs.coordinates = input_coordinates

        return jbm_inputs

    def pred_transform(self, text_arr, coords_list=[]):
        input_ids = []
        input_mask = []
        segment_ids = []
        valid_positions = []
        input_coordinates = []
        
        for i in range(len(text_arr)):
            text = text_arr[i]
            coords = []
            if coords_list:
                coords = coords_list[i]
            ids, mask, seg_ids, valid_pos, coords = self.__vectorize(text, coords)
            if len(ids) > 512:
                for j in range(0, len(ids)-511, 50):
                    input_ids.append(ids[j:j+512])
                    input_mask.append(mask[j:j+512])
                    segment_ids.append(seg_ids[j:j+512])
                    valid_positions.append(valid_pos[j:j+512])
                    if coords:
                        input_coordinates.append(coords[j:j+512])
                if len(ids)-512 % 50 != 0:
                    input_ids.append(ids[len(ids)-512:len(ids)])
                    input_mask.append(mask[len(ids)-512:len(ids)])
                    segment_ids.append(seg_ids[len(ids)-512:len(ids)])
                    valid_positions.append(valid_pos[len(ids)-512:len(ids)])
                    if coords:
                        input_coordinates.append(coords[len(ids)-512:len(ids)])
            else:                    
                input_ids.append(ids)
                input_mask.append(mask)
                segment_ids.append(seg_ids)
                valid_positions.append(valid_pos)
                if coords:
                    input_coordinates.append(coords)

        ## set the maximum length is 50
        input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids, maxlen=512, truncating='post', padding='post')
        input_mask = tf.keras.preprocessing.sequence.pad_sequences(input_mask, maxlen=512, truncating='post', padding='post')
        segment_ids = tf.keras.preprocessing.sequence.pad_sequences(segment_ids, maxlen=512, truncating='post', padding='post')
        valid_positions = tf.keras.preprocessing.sequence.pad_sequences(valid_positions, maxlen=512, truncating='post', padding='post')

        if coords_list:
            for i in range(len(input_coordinates)):
                if len(input_coordinates[i]) > 512:
                    input_coordinates[i] = input_coordinates[i][:512]
                else:
                    input_coordinates[i].extend([[0.] * 4] * (512 - len(input_coordinates[i])))
            input_coordinates = np.array(input_coordinates)

            return input_ids, input_mask, segment_ids, valid_positions, input_coordinates

        return input_ids, input_mask, segment_ids, valid_positions

    def test_transform(self, text_arr, tags_arr, intents):
        input_ids = []
        input_mask = []
        segment_ids = []
        valid_positions = []
        tags_arr_2 = []
        intents_2 = []
        
        for i in range(len(text_arr)):
            text, tags, intent = text_arr[i], tags_arr[i], intents[i]
            tags = tags.split()
            ids, mask, seg_ids, valid_pos = self.__vectorize(text)
            if len(ids) > 512:
                for i in range(0, len(ids)-511, 50):
                    input_ids.append(ids[i:i+512])
                    input_mask.append(mask[i:i+512])
                    segment_ids.append(seg_ids[i:i+512])
                    valid_positions.append(valid_pos[i:i+512])
                    tags_arr_2.append(tags[:sum(valid_pos[i:i+512])])
                    tags = tags[sum(valid_pos[i:i+512]):]
                    intents_2.append(intent)
                if len(ids)-512 % 50 != 0:
                    input_ids.append(ids[len(ids)-512:len(ids)])
                    input_mask.append(mask[len(ids)-512:len(ids)])
                    segment_ids.append(seg_ids[len(ids)-512:len(ids)])
                    valid_positions.append(valid_pos[len(ids)-512:len(ids)])
                    tags_arr_2.append(tags[:sum(valid_pos[len(ids)-512:len(ids)])])
                    tags = tags[sum(valid_pos[len(ids)-512:len(ids)]):]
                    intents_2.append(intent)
            else:                    
                input_ids.append(ids)
                input_mask.append(mask)
                segment_ids.append(seg_ids)
                valid_positions.append(valid_pos)
                tags_arr_2.append(tags)
                intents_2.append(intent)

        sequence_length = np.array([len(i) for i in input_ids])

        ## set the maximum length is 50
        input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids, maxlen=512, truncating='post', padding='post')
        input_mask = tf.keras.preprocessing.sequence.pad_sequences(input_mask, maxlen=512, truncating='post', padding='post')
        segment_ids = tf.keras.preprocessing.sequence.pad_sequences(segment_ids, maxlen=512, truncating='post', padding='post')
        valid_positions = tf.keras.preprocessing.sequence.pad_sequences(valid_positions, maxlen=512, truncating='post', padding='post')

        return input_ids, input_mask, segment_ids, valid_positions, sequence_length, tags_arr_2, intents_2