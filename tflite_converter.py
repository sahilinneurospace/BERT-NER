import tensorflow as tf
#from joint_bert_model import sparse_categorical_focal_loss
from bert_model import BertLayer 
tflite_converter = tf.lite.TFLiteConverter.from_keras_model_file('saved_models/slot_tagger_intent_classifier_small_L-2_H-128_A-2_all/joint_bert_model.h5', custom_objects = {'BertLayer' : BertLayer, 'sparse_categorical_focal_loss':sparse_categorical_focal_loss}, input_shapes = {'input_ids': [None, 512], 'input_masks': [None, 512], 'segment_ids': [None, 512], 'valid_positions': [None, 512, 24]})
tflite_converter.experimental_new_converter = True 
tflite_converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS] 
tflite_model = tflite_converter.convert()
with tf.io.gfile.GFile('saved_models/slot_tagger_intent_classifier_small_L-2_H-128_A-2_all/joint_bert_model.tflite', 'wb') as f: 
	f.write(tflite_model)