from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import math
import six
import tensorflow as tf
from tensorflow.python.keras import backend as K
import keras

'''
Custom Layer to concatenate the CLS token vector to each token's vector in the corresponding sentence.
'''
class CLSProcess(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.state_size = units
        super(CLSProcess, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, states):
        x = K.expand_dims(tf.unstack(inputs, axis=-1)[0], axis=1)
        y = K.expand_dims(tf.unstack(inputs, axis=-1)[1], axis=1)
        z = K.stack(tf.unstack(inputs, axis=-1)[2:], axis=-1)
        prev_output = states[0]
        output = y * prev_output + x * z
        return output, [output]

def gelu(x):
  return 0.5 * x * (1 + tf.math.tanh((2/math.pi)**0.5 * (x + 0.044715 * x**3)))

class Attention(tf.keras.layers.Layer):
  """Performs multi-headed attention from `from_tensor` to `to_tensor`.

  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `from_tensor` and `to_tensor` are the same, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-with vector.

  This function first projects `from_tensor` into a "query" tensor and
  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
  of tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].

  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor and returned.

  In practice, the multi-headed attention are done with tf.einsum as follows:
    Input_tensor: [BFD]
    Wq, Wk, Wv: [DNH]
    Q:[BFNH] = einsum('BFD,DNH->BFNH', Input_tensor, Wq)
    K:[BTNH] = einsum('BTD,DNH->BTNH', Input_tensor, Wk)
    V:[BTNH] = einsum('BTD,DNH->BTNH', Input_tensor, Wv)
    attention_scores:[BNFT] = einsum('BTNH,BFNH->BNFT', K, Q) / sqrt(H)
    attention_probs:[BNFT] = softmax(attention_scores)
    context_layer:[BFNH] = einsum('BNFT,BTNH->BFNH', attention_probs, V)
    Wout:[DNH]
    Output:[BFD] = einsum('BFNH,DNH>BFD', context_layer, Wout)
  """

  def __init__(self,
               num_attention_heads=12,
               size_per_head=64,
               attention_probs_dropout_prob=0.0,
               initializer_range=0.02,
               backward_compatible=False,
               **kwargs):
    super(Attention, self).__init__(**kwargs)
    self.num_attention_heads = num_attention_heads
    self.size_per_head = size_per_head
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer_range = initializer_range
    self.backward_compatible = backward_compatible

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.query_dense = self._projection_dense_layer("query")
    self.key_dense = self._projection_dense_layer("key")
    self.value_dense = self._projection_dense_layer("value")
    self.attention_probs_dropout = tf.keras.layers.Dropout(
        rate=self.attention_probs_dropout_prob)
    super(Attention, self).build(unused_input_shapes)

  def reshape_to_matrix(self, input_tensor):
    """Reshape N > 2 rank tensor to rank 2 tensor for performance."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
      raise ValueError("Input tensor must have at least rank 2."
                       "Shape = %s" % (input_tensor.shape))
    if ndims == 2:
      return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor

  def __call__(self, from_tensor, to_tensor, attention_mask=None, **kwargs):
    inputs = [from_tensor, to_tensor, attention_mask]
    return super(Attention, self).__call__(inputs, **kwargs)

  def call(self, inputs):
    """Implements call() for the layer."""
    from_tensor, to_tensor, attention_mask = inputs[0], inputs[1], inputs[2]

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`
    # `query_tensor` = [B, F, N ,H]
    query_tensor = self.query_dense(from_tensor)

    # `key_tensor` = [B, T, N, H]
    key_tensor = self.key_dense(to_tensor)

    # `value_tensor` = [B, T, N, H]
    value_tensor = self.value_dense(to_tensor)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    attention_scores = tf.einsum("BTNH,BFNH->BNFT", key_tensor, query_tensor)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(self.size_per_head)))

    if attention_mask is not None:
      # `attention_mask` = [B, 1, F, T]
      attention_mask = tf.expand_dims(attention_mask, axis=[1])

      # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and -10000.0 for masked positions.
      adder = (1.0 - tf.cast(attention_mask, attention_scores.dtype)) * -10000.0

      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      attention_scores += adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.attention_probs_dropout(attention_probs)

    # `context_layer` = [B, F, N, H]
    context_tensor = tf.einsum("BNFT,BTNH->BFNH", attention_probs, value_tensor)

    return context_tensor

  def _projection_dense_layer(self, name):
    """A helper to define a projection layer."""
    return Dense3D(
        num_attention_heads=self.num_attention_heads,
        size_per_head=self.size_per_head,
        kernel_initializer=get_initializer(self.initializer_range),
        output_projection=False,
        backward_compatible=self.backward_compatible,
        name=name)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'num_attention_heads': self.num_attention_heads,
        'size_per_head': self.size_per_head,
        'attention_probs_dropout_prob': self.attention_probs_dropout_prob,
        'initializer_range': self.initializer_range,
        'backward_compatible': self.backward_compatible
    })
    return config


class Dense3D(tf.keras.layers.Layer):
  """A Dense Layer using 3D kernel with tf.einsum implementation.

  Attributes:
    num_attention_heads: An integer, number of attention heads for each
      multihead attention layer.
    size_per_head: An integer, hidden size per attention head.
    hidden_size: An integer, dimension of the hidden layer.
    kernel_initializer: An initializer for the kernel weight.
    bias_initializer: An initializer for the bias.
    activation: An activation function to use. If nothing is specified, no
      activation is applied.
    use_bias: A bool, whether the layer uses a bias.
    output_projection: A bool, whether the Dense3D layer is used for output
      linear projection.
    backward_compatible: A bool, whether the variables shape are compatible
      with checkpoints converted from TF 1.x.
  """

  def __init__(self,
               num_attention_heads=12,
               size_per_head=72,
               kernel_initializer=None,
               bias_initializer="zeros",
               activation=None,
               use_bias=True,
               output_projection=False,
               backward_compatible=False,
               **kwargs):
    """Inits Dense3D."""
    super(Dense3D, self).__init__(**kwargs)
    self.num_attention_heads = num_attention_heads
    self.size_per_head = size_per_head
    self.hidden_size = num_attention_heads * size_per_head
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.activation = activation
    self.use_bias = use_bias
    self.output_projection = output_projection
    self.backward_compatible = backward_compatible

  @property
  def compatible_kernel_shape(self):
    if self.output_projection:
      return [self.hidden_size, self.hidden_size]
    return [self.last_dim, self.hidden_size]

  @property
  def compatible_bias_shape(self):
    return [self.hidden_size]

  @property
  def kernel_shape(self):
    if self.output_projection:
      return [self.num_attention_heads, self.size_per_head, self.hidden_size]
    return [self.last_dim, self.num_attention_heads, self.size_per_head]

  @property
  def bias_shape(self):
    if self.output_projection:
      return [self.hidden_size]
    return [self.num_attention_heads, self.size_per_head]

  def build(self, input_shape):
    """Implements build() for the layer."""
    dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError("Unable to build `Dense3D` layer with non-floating "
                      "point (and non-complex) dtype %s" % (dtype,))
    input_shape = tf.TensorShape(input_shape)
    if tf.compat.dimension_value(input_shape[-1]) is None:
      raise ValueError("The last dimension of the inputs to `Dense3D` "
                       "should be defined. Found `None`.")
    self.last_dim = tf.compat.dimension_value(input_shape[-1])
    self.input_spec = tf.keras.layers.InputSpec(
        min_ndim=3, axes={-1: self.last_dim})
    # Determines variable shapes.
    if self.backward_compatible:
      kernel_shape = self.compatible_kernel_shape
      bias_shape = self.compatible_bias_shape
    else:
      kernel_shape = self.kernel_shape
      bias_shape = self.bias_shape

    self.kernel = self.add_weight(
        "kernel",
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        dtype=self.dtype,
        trainable=True)
    if self.use_bias:
      self.bias = self.add_weight(
          "bias",
          shape=bias_shape,
          initializer=self.bias_initializer,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None
    super(Dense3D, self).build(input_shape)

  def call(self, inputs):
    """Implements ``call()`` for Dense3D.

    Args:
      inputs: A float tensor of shape [batch_size, sequence_length, hidden_size]
        when output_projection is False, otherwise a float tensor of shape
        [batch_size, sequence_length, num_heads, dim_per_head].

    Returns:
      The projected tensor with shape [batch_size, sequence_length, num_heads,
        dim_per_head] when output_projection is False, otherwise [batch_size,
        sequence_length, hidden_size].
    """
    if self.backward_compatible:
      kernel = tf.keras.backend.reshape(self.kernel, self.kernel_shape)
      bias = (tf.keras.backend.reshape(self.bias, self.bias_shape)
              if self.use_bias else None)
    else:
      kernel = self.kernel
      bias = self.bias

    if self.output_projection:
      ret = tf.einsum("abcd,cde->abe", inputs, kernel)
    else:
      ret = tf.einsum("abc,cde->abde", inputs, kernel)
    if self.use_bias:
      ret += bias
    if self.activation is not None:
      return self.activation(ret)
    return ret

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'num_attention_heads': self.num_attention_heads,
        'size_per_head': self.size_per_head,
        'hidden_size': self.hidden_size,
        'kernel_initializer': self.kernel_initializer,
        'bias_initializer': self.bias_initializer,
        'activation': self.activation,
        'use_bias': self.use_bias,
        'output_projection': self.output_projection,
        'backward_compatible': self.backward_compatible
    })
    return config


class Dense2DProjection(tf.keras.layers.Layer):
  """A 2D projection layer with tf.einsum implementation."""

  def __init__(self,
               output_size,
               kernel_initializer=None,
               bias_initializer="zeros",
               activation=None,
               fp32_activation=False,
               **kwargs):
    super(Dense2DProjection, self).__init__(**kwargs)
    self.output_size = output_size
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.activation = activation
    self.fp32_activation = fp32_activation

  def build(self, input_shape):
    """Implements build() for the layer."""
    dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError("Unable to build `Dense2DProjection` layer with "
                      "non-floating point (and non-complex) "
                      "dtype %s" % (dtype,))
    input_shape = tf.TensorShape(input_shape)
    if tf.compat.dimension_value(input_shape[-1]) is None:
      raise ValueError("The last dimension of the inputs to "
                       "`Dense2DProjection` should be defined. "
                       "Found `None`.")
    last_dim = tf.compat.dimension_value(input_shape[-1])
    self.input_spec = tf.keras.layers.InputSpec(min_ndim=3, axes={-1: last_dim})
    self.kernel = self.add_weight(
        "kernel",
        shape=[last_dim, self.output_size],
        initializer=self.kernel_initializer,
        dtype=self.dtype,
        trainable=True)
    self.bias = self.add_weight(
        "bias",
        shape=[self.output_size],
        initializer=self.bias_initializer,
        dtype=self.dtype,
        trainable=True)
    super(Dense2DProjection, self).build(input_shape)

  def call(self, inputs):
    """Implements call() for Dense2DProjection.

    Args:
      inputs: float Tensor of shape [batch, from_seq_length,
        num_attention_heads, size_per_head].

    Returns:
      A 3D Tensor.
    """
    ret = tf.einsum("abc,cd->abd", inputs, self.kernel)
    ret += self.bias
    if self.activation is not None:
      if self.dtype == tf.float16 and self.fp32_activation:
        ret = tf.cast(ret, tf.float32)
      print(self.activation, "AAAAAAAAAAAA")
      return gelu(ret)
    return ret

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'output_size' : self.output_size,
        'kernel_initializer' : self.kernel_initializer,
        'bias_initializer' : self.bias_initializer,
        'activation' : self.activation,
        'fp32_activation' : self.fp32_activation
    })
    return config


class TransformerBlock(tf.keras.layers.Layer):
  """Single transformer layer.

  It has two sub-layers. The first is a multi-head self-attention mechanism, and
  the second is a positionwise fully connected feed-forward network.
  """

  def __init__(self,
               hidden_size=768,
               num_attention_heads=12,
               intermediate_size=3072,
               intermediate_activation=gelu,
               hidden_dropout_prob=0.0,
               attention_probs_dropout_prob=0.0,
               initializer_range=0.02,
               backward_compatible=False,
               float_type=tf.float32,
               **kwargs):
    super(TransformerBlock, self).__init__(**kwargs)
    self.hidden_size = hidden_size
    self.num_attention_heads = num_attention_heads
    self.intermediate_size = intermediate_size
    self.intermediate_activation = intermediate_activation
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer_range = initializer_range
    self.backward_compatible = backward_compatible
    self.float_type = float_type

    if self.hidden_size % self.num_attention_heads != 0:
      raise ValueError(
          "The hidden size (%d) is not a multiple of the number of attention "
          "heads (%d)" % (self.hidden_size, self.num_attention_heads))
    self.attention_head_size = int(self.hidden_size / self.num_attention_heads)

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.attention_layer = Attention(
        num_attention_heads=self.num_attention_heads,
        size_per_head=self.attention_head_size,
        attention_probs_dropout_prob=self.attention_probs_dropout_prob,
        initializer_range=self.initializer_range,
        backward_compatible=self.backward_compatible,
        name="self_attention")
    self.attention_output_dense = Dense3D(
        num_attention_heads=self.num_attention_heads,
        size_per_head=int(self.hidden_size / self.num_attention_heads),
        kernel_initializer=get_initializer(self.initializer_range),
        output_projection=True,
        backward_compatible=self.backward_compatible,
        name="self_attention_output")
    self.attention_dropout = tf.keras.layers.Dropout(
        rate=self.hidden_dropout_prob)
    self.attention_layer_norm = (
        tf.keras.layers.LayerNormalization(
            name="self_attention_layer_norm", axis=-1, epsilon=1e-12,
            # We do layer norm in float32 for numeric stability.
            dtype=tf.float32))
    self.intermediate_dense = Dense2DProjection(
        output_size=self.intermediate_size,
        kernel_initializer=get_initializer(self.initializer_range),
        activation=self.intermediate_activation,
        # Uses float32 so that gelu activation is done in float32.
        fp32_activation=True,
        name="intermediate")
    self.output_dense = Dense2DProjection(
        output_size=self.hidden_size,
        kernel_initializer=get_initializer(self.initializer_range),
        name="output")
    self.output_dropout = tf.keras.layers.Dropout(rate=self.hidden_dropout_prob)
    self.output_layer_norm = tf.keras.layers.LayerNormalization(
        name="output_layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32)
    super(TransformerBlock, self).build(unused_input_shapes)

  def common_layers(self):
    """Explicitly gets all layer objects inside a Transformer encoder block."""
    return [
        self.attention_layer, self.attention_output_dense,
        self.attention_dropout, self.attention_layer_norm,
        self.intermediate_dense, self.output_dense, self.output_dropout,
        self.output_layer_norm
    ]

  def __call__(self, input_tensor, attention_mask=None, **kwargs):
    inputs = [input_tensor, attention_mask]
    return super(TransformerBlock, self).__call__(inputs, **kwargs)

  def call(self, inputs):
    """Implements call() for the layer."""
    input_tensor, attention_mask = inputs[0], inputs[1]
    attention_output = self.attention_layer(
        from_tensor=input_tensor,
        to_tensor=input_tensor,
        attention_mask=attention_mask)
    attention_output = self.attention_output_dense(attention_output)
    attention_output = self.attention_dropout(attention_output)
    # Use float32 in keras layer norm and the gelu activation in the
    # intermediate dense layer for numeric stability
    attention_output = self.attention_layer_norm(input_tensor +
                                                 attention_output)
    if self.float_type == tf.float16:
      attention_output = tf.cast(attention_output, tf.float16)
    intermediate_output = self.intermediate_dense(attention_output)
    if self.float_type == tf.float16:
      intermediate_output = tf.cast(intermediate_output, tf.float16)
    layer_output = self.output_dense(intermediate_output)
    layer_output = self.output_dropout(layer_output)
    # Use float32 in keras layer norm for numeric stability
    layer_output = self.output_layer_norm(layer_output + attention_output)
    if self.float_type == tf.float16:
      layer_output = tf.cast(layer_output, tf.float16)
    return layer_output

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'hidden_size': self.hidden_size,
        'num_attention_heads': self.num_attention_heads,
        'intermediate_size': self.intermediate_size,
        'intermediate_activation': self.intermediate_activation,
        'hidden_dropout_prob': self.hidden_dropout_prob,
        'attention_probs_dropout_prob': self.attention_probs_dropout_prob,
        'initializer_range': self.initializer_range,
        'backward_compatible': self.backward_compatible,
        'float_type': self.float_type
    })
    return config

class Transformer(tf.keras.layers.Layer):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
  """

  def __init__(self,
               num_hidden_layers=12,
               hidden_size=768,
               num_attention_heads=12,
               intermediate_size=3072,
               intermediate_activation=gelu,
               hidden_dropout_prob=0.0,
               attention_probs_dropout_prob=0.0,
               initializer_range=0.02,
               backward_compatible=False,
               float_type=tf.float32,
               **kwargs):
    super(Transformer, self).__init__(**kwargs)
    self.num_hidden_layers = num_hidden_layers
    self.hidden_size = hidden_size
    self.num_attention_heads = num_attention_heads
    self.intermediate_size = intermediate_size
    self.intermediate_activation = intermediate_activation
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer_range = initializer_range
    self.backward_compatible = backward_compatible
    self.float_type = float_type

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.layers = []
    for i in range(self.num_hidden_layers):
      self.layers.append(
          TransformerBlock(
              hidden_size=self.hidden_size,
              num_attention_heads=self.num_attention_heads,
              intermediate_size=self.intermediate_size,
              intermediate_activation=self.intermediate_activation,
              hidden_dropout_prob=self.hidden_dropout_prob,
              attention_probs_dropout_prob=self.attention_probs_dropout_prob,
              initializer_range=self.initializer_range,
              backward_compatible=self.backward_compatible,
              float_type=self.float_type,
              name=("layer_%d" % i)))
    super(Transformer, self).build(unused_input_shapes)

  def call(self, x, return_all_layers=False):
    """Implements call() for the layer.

    Args:
      inputs: packed inputs.
      return_all_layers: bool, whether to return outputs of all layers inside
        encoders.
    Returns:
      Output tensor of the last layer or a list of output tensors.
    """
    input_tensor = x
    output_tensor = input_tensor

    all_layer_outputs = []
    for layer in self.layers:
      output_tensor = layer(output_tensor, None)
      all_layer_outputs.append(output_tensor)

    if return_all_layers:
      return all_layer_outputs

    return all_layer_outputs[-1]

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'num_hidden_layers': self.num_hidden_layers,
        'hidden_size': self.hidden_size,
        'num_attention_heads': self.num_attention_heads,
        'intermediate_size': self.intermediate_size,
        'intermediate_activation': self.intermediate_activation,
        'hidden_dropout_prob': self.hidden_dropout_prob,
        'attention_probs_dropout_prob': self.attention_probs_dropout_prob,
        'initializer_range': self.initializer_range,
        'backward_compatible': self.backward_compatible,
        'float_type': self.float_type
    })
    return config


def get_initializer(initializer_range=0.02):
  """Creates a `tf.initializers.truncated_normal` with the given range.

  Args:
    initializer_range: float, initializer range for stddev.

  Returns:
    TruncatedNormal initializer with stddev = `initializer_range`.
  """
  return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)
