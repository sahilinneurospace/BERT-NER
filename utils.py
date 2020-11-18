import dateparser
from spellchecker import SpellChecker
import os
import json
from tags_vectorizer import *

class RawData():
    def __init__(self, text=[], tags=[], coordinates=[], intents=[]):
        self.text = text
        self.tags = tags
        self.coordinates = coordinates
        self.intents = intents

class ModelFiles():
    def __init__(self, model=None, intents_label_encoder=None, tags_vectorizer=None):
        self.model = model
        self.intents_label_encoder = intents_label_encoder
        self.tags_vectorizer = tags_vectorizer

class BertInputs():
    def __init__(self, input_ids=None, input_mask=None, segment_ids=None, valid_positions=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.valid_positions = valid_positions

#joint bert model inputs
class JointBertModelInputs():
    def __init__(self, bert_inputs=None, coordinates=None, cls_sep_markers=None, sequence_length=None):
        self.bert_inputs = bert_inputs
        self.input_ids = bert_inputs.input_ids
        self.input_mask = bert_inputs.input_mask
        self.segment_ids = bert_inputs.segment_ids
        self.valid_positions = bert_inputs.valid_positions
        self.coordinates = coordinates
        self.cls_sep_markers = cls_sep_markers
        self.sequence_length = sequence_length

def DateTimeParser(date, time):
    x = dateparser.parse(date)
    if x is None:
        spell = SpellChecker()
        date = date.split()
        x = []
        for tok in date:
            x.append(spell.correction(tok))
        if x:
            x = " ".join(x)
        else:
            x = ""
        x = dateparser.parse(x)
    date = x
    if date:
        date = str(date)[:10]
    else:
        date = "0000-00-00"
    x = dateparser.parse(time)
    if x is None:
        spell = SpellChecker()
        time = time.split()
        x = []
        for tok in time:
            x.append(spell.correction(tok))
        if x:
            print(x)
            x = " ".join(x)
        else:
            x = ""
        x = dateparser.parse(x)
    time = x
    if time:
        time = str(time)[11:]
    else:
        time = "00:00:00"
    return date, time


def load_intents_label_encoder(load_folder_path):
    
    with open(os.path.join(load_folder_path, 'intents_label_encoder.json'), 'r') as f:
        intents_list = json.load(f)
        intents_num = len(intents_list)
        intents_label_encoder = LabelEncoder()
        intents_label_encoder.tags = intents_list
    
    return intents_label_encoder, intents_num


def load_tags_vectorizer(load_folder_path, multiNER, intents_num=1):

    if not multiNER:
        with open(os.path.join(load_folder_path, 'tags_vectorizer.json'), 'r') as f:
            tags_list = json.load(f)
            slots_num = len(tags_list)
            tags_vectorizer = TagsVectorizer()
            tags_vectorizer.label_encoder.tags = tags_list
    else:
        slots_num = []
        tags_vectorizers = []
        for i in range(intents_num):
            with open(os.path.join(load_folder_path, 'tags_vectorizer_{i}.json'), 'r') as f:
                tags_list = json.load(f)
                tags_vectorizer = TagsVectorizer()
                tags_vectorizer.label_encoder.tags = tags_list
                tags_vectorizers.append(tags_vectorizer)
                slots_num.append(len(tags_list))
        tags_vectorizer = tags_vectorizers

    return tags_vectorizer, slots_num
