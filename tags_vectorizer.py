import numpy as np

class LabelEncoder:
    
    def __init__(self) :
        self.tags = []

    def fit(self, data):
        data = list(set(data))
        self.tags = data

    def transform(self, seq):
        assert isinstance(seq, list)
        assert not [x for x in seq if x not in self.tags]
        return np.array([self.tags.index(x) for x in seq])

    def inverse_transform(self, seq):
        assert isinstance(seq, list)
        assert max(seq) < len(self.tags)
        return [self.tags[x] for x in seq]

class TagsVectorizer:

    def __init__(self):
        self.label_encoder = LabelEncoder()

    def tokenize(self, tags_str_arr):
        return [s.split() for s in tags_str_arr]

    def fit(self, train_tags_str_arr, val_tags_str_arr):
        ## in order to avoid, in valid_dataset, there is tags which not exit in train_dataset. like: ATIS datset
        data = ["[padding]", "[CLS]", "[SEP]"] + [item for sublist in self.tokenize(train_tags_str_arr) for item in sublist]
        data = data + [item for sublist in self.tokenize(val_tags_str_arr) for item in sublist]
        ## # data:  ["[padding]", "[CLS]", "[SEP]", all of the real tags]; add the "[padding]", "[CLS]", "[SEP]" for the real tag list
        self.label_encoder.fit(data)

    def transform(self, tags_str_arr, valid_positions):
        ## if we set the maximum length is 50, then the seq_length is 50; otherwise, it will be equal to the maximal length of dataset
        if isinstance(valid_positions, list):
            if len(valid_positions):
                seq_length = len(valid_positions[0])
            else:
                seq_length = 0
        else:
            seq_length = valid_positions.shape[1] # .shape[0]: number of rows, .shape[1]: number of columns
        data = self.tokenize(tags_str_arr)
        ## we added the 'CLS' and 'SEP' token as the first and last token for every sentence respectively
        data = [self.label_encoder.transform(["[CLS]"] + x + ["[SEP]"]).astype(np.int32) for x in data] #upper 'O', not 0

        output = np.zeros((len(data), seq_length))
        for i in range(len(data)):
            idx = 0
            for j in range(seq_length):
                if valid_positions[i][j] == 1:
                    output[i][j] = data[i][idx]
                    idx += 1
                else:
                    output[i][j] = -1
        return output


    def inverse_transform(self, model_output_3d, valid_position):
        ## model_output_3d is the predicted slots output of trained model
        seq_length = valid_position.shape[1]
        slots = np.argmax(model_output_3d, axis=-1)
        slots = [self.label_encoder.inverse_transform(y.tolist()) for y in slots]
        output = []
        for i in range(len(slots)):
            y = []
            for j in range(seq_length):
                if valid_position[i][j] == 1: ## only valid_positions = 1 have the real slot-tag
                    y.append(str(slots[i][j]))
            output.append(y)
        return output

    def load(self):
        pass

    def save(self):
        pass

def get_cls_sep_markers(text_str_arr, valid_positions):
    ## if we set the maximum length is 50, then the seq_length is 50; otherwise, it will be equal to the maximal length of dataset
    if isinstance(valid_positions, list):
        if len(valid_positions):
            seq_length = len(valid_positions[0])
        else:
            seq_length = 0
    else:
        seq_length = valid_positions.shape[1] # .shape[0]: number of rows, .shape[1]: number of columns
    data = [s.split() for s in text_str_arr]
    ## we added the 'CLS' and 'SEP' token as the first and last token for every sentence respectively
    data = [["[CLS]"] + x + ["[SEP]"] for x in data] #upper 'O', not 0

    cls_sep_markers = []
    for i in range(len(data)):
        marker = [np.array([0, 0])]
        idx = 1
        for j in range(1, seq_length):
            if valid_positions[i][j] == 1:
                if data[i][idx] == '[CLS]':
                    marker.append(np.array([1, 0]))
                elif data[i][idx] == '[SEP]':
                    marker.append(np.array([0, 0]))
                else:
                    marker.append(np.array([0, 1]))
                idx += 1
            else:
                marker.append(np.array([0, 1]))

        cls_sep_markers.append(np.array(marker))

    return np.array(cls_sep_markers)


if __name__ == '__main__':
    train_tags_str_arr = ['O O B-X B-Y', 'O B-Y O']
    val_tags_str_arr = ['O O B-X B-Y', 'O B-Y O XXX']
    valid_positions = np.array([[1, 1, 1, 1, 0, 1, 1], [1, 1, 0, 1, 1, 0, 1]])

    vectorizer = TagsVectorizer()
    vectorizer.fit(train_tags_str_arr, val_tags_str_arr)
    data = vectorizer.transform(train_tags_str_arr, valid_positions)
    print(data, vectorizer.label_encoder.classes_)