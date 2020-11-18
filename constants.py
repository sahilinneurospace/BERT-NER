OUTSIDE_ENTITY_LABEL = 'O'
BERT_CLS_TOK = '[CLS]'
BERT_SEP_TOK = '[SEP]'
BEGIN_ENTITY_PREFIX = 'B_'

# exhaustive list of NER entities currently in use
ENTITY_LIST = ['Movie', 'Book', 'Genre', 'ToDo', 'O', 'Quantity',
               'Email', 'Phone', 'Time', 'Event', 'Date',
               'Address', 'Details',  'Url', 'Item', 'Subevent']

# intent to architecture mapping set according to optimal performance evaluation
# among all architectures for each intent
intentwise_architecture = {'Shopping(Grocery)': 3, 'Event': 4}

## specifications corresponding to each single-intent architecture
architecture_specs = {
    1: {'intents_dnn_depth': 1, 'entities_dnn_depth': 5, 'transformer_block_depth': 0, 'use_layout_data': False, 'use_cls': False},
    2: {'intents_dnn_depth': 1, 'entities_dnn_depth': 5, 'transformer_block_depth': 0, 'use_layout_data': True, 'use_cls': False},
    3: {'intents_dnn_depth': 1, 'entities_dnn_depth': 4, 'transformer_block_depth': 1, 'use_layout_data': True, 'use_cls': False},
    4: {'intents_dnn_depth': 1, 'entities_dnn_depth': 3, 'transformer_block_depth': 1, 'use_layout_data': True, 'use_cls': True}
}