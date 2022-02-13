# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The policy and value networks share a majority of their architecture.
This helps the intermediate layers extract concepts that are relevant to both
move prediction and score estimation.
"""

from pyexpat import model
from re import T
from absl import flags, app
import functools
import json
import logging
import os.path
import struct
import tempfile
import time
import numpy as np
import random

import tensorflow as tf
# from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
# from tensorflow.contrib import quantize as contrib_quantize
# from tensorflow.contrib import summary as contrib_summary
# from tensorflow.contrib import tpu as contrib_tpu
# from tensorflow.contrib.tpu.python.tpu import tpu_config as contrib_tpu_python_tpu_tpu_config
# from tensorflow.contrib.tpu.python.tpu import tpu_estimator as contrib_tpu_python_tpu_tpu_estimator
# from tensorflow.contrib.tpu.python.tpu import tpu_optimizer as contrib_tpu_python_tpu_tpu_optimizer


import features as features_lib
import go


flags.DEFINE_integer('train_batch_size', 256,
                     'Batch size to use for train/eval evaluation. For GPU '
                     'this is batch size as expected. If \"use_tpu\" is set,'
                     'final batch size will be = train_batch_size * num_tpu_cores')

flags.DEFINE_integer('conv_width', 256 if go.N == 19 else 32,
                     'The width of each conv layer in the shared trunk.')

flags.DEFINE_integer('policy_conv_width', 2,
                     'The width of the policy conv layer.')

flags.DEFINE_integer('value_conv_width', 1,
                     'The width of the value conv layer.')

flags.DEFINE_integer('fc_width', 256 if go.N == 19 else 64,
                     'The width of the fully connected layer in value head.')

flags.DEFINE_integer('trunk_layers', go.N,
                     'The number of resnet layers in the shared trunk.')

flags.DEFINE_multi_integer('lr_boundaries', [400000, 600000],
                           'The number of steps at which the learning rate will decay')

flags.DEFINE_multi_float('lr_rates', [0.01, 0.001, 0.0001],
                         'The different learning rates')

flags.DEFINE_integer('training_seed', 0,
                     'Random seed to use for training and validation')

flags.register_multi_flags_validator(
    ['lr_boundaries', 'lr_rates'],
    lambda flags: len(flags['lr_boundaries']) == len(flags['lr_rates']) - 1,
    'Number of learning rates must be exactly one greater than the number of boundaries')

flags.DEFINE_float('l2_strength', 1e-4,
                   'The L2 regularization parameter applied to weights.')

flags.DEFINE_float('value_cost_weight', 1.0,
                   'Scalar for value_cost, AGZ paper suggests 1/100 for '
                   'supervised learning')

flags.DEFINE_float('sgd_momentum', 0.9,
                   'Momentum parameter for learning rate.')

flags.DEFINE_string('work_dir', None,
                    'The Estimator working directory. Used to dump: '
                    'checkpoints, tensorboard logs, etc..')

flags.DEFINE_bool('use_tpu', False, 'Whether to use TPU for training.')

flags.DEFINE_string(
    'tpu_name', None,
    'The Cloud TPU to use for training. This should be either the name used'
    'when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')

flags.DEFINE_integer(
    'num_tpu_cores', default=8,
    help=('Number of TPU cores. For a single TPU device, this is 8 because each'
          ' TPU has 4 chips each with 2 cores.'))

flags.DEFINE_string('gpu_device_list', None,
                    'Comma-separated list of GPU device IDs to use.')

flags.DEFINE_bool('quantize', False,
                  'Whether create a quantized model. When loading a model for '
                  'inference, this must match how the model was trained.')

flags.DEFINE_integer('quant_delay', 700 * 1024,
                     'Number of training steps after which weights and '
                     'activations are quantized.')

flags.DEFINE_integer(
    'iterations_per_loop', 128,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))

flags.DEFINE_integer(
    'summary_steps', default=256,
    help='Number of steps between logging summary scalars.')

flags.DEFINE_integer(
    'keep_checkpoint_max', default=5, help='Number of checkpoints to keep.')

flags.DEFINE_bool(
    'use_random_symmetry', True,
    help='If true random symmetries be used when doing inference.')

flags.DEFINE_bool(
    'use_SE', False,
    help='Use Squeeze and Excitation.')

flags.DEFINE_bool(
    'use_SE_bias', False,
    help='Use Squeeze and Excitation with bias.')

flags.DEFINE_integer(
    'SE_ratio', 2,
    help='Squeeze and Excitation ratio.')

flags.DEFINE_bool(
    'use_swish', False,
    help=('Use Swish activation function inplace of ReLu. '
          'https://arxiv.org/pdf/1710.05941.pdf'))

flags.DEFINE_bool(
    'bool_features', False,
    help='Use bool input features instead of float')

flags.DEFINE_string(
    'input_features', 'agz',
    help='Type of input features: "agz" or "mlperf07"')

flags.DEFINE_string(
    'input_layout', 'nhwc',
    help='Layout of input features: "nhwc" or "nchw"')


# TODO(seth): Verify if this is still required.
flags.register_multi_flags_validator(
    ['use_tpu', 'iterations_per_loop', 'summary_steps'],
    lambda flags: (not flags['use_tpu'] or
                   flags['summary_steps'] % flags['iterations_per_loop'] == 0),
    'If use_tpu, summary_steps must be a multiple of iterations_per_loop')

FLAGS = flags.FLAGS

def get_features():
    if FLAGS.input_features == 'agz':
        return features_lib.AGZ_FEATURES
    elif FLAGS.input_features == 'mlperf07':
        return features_lib.MLPERF07_FEATURES
    else:
        raise ValueError('unrecognized input features "%s"' %
                         FLAGS.input_features)



class DualNetwork(tf.keras.Model):
    def __init__(self, save_file, params=FLAGS.flag_values_dict()):
        super().__init__()
        self.save_file = save_file
        self.inference_fn = ModelInferenceFn(params=params)

    def call(self, inputs):
        return self.inference_fn(inputs)

    def run_many(self, positions):
        f = get_features()
        processed = [features_lib.extract_features(p, f) for p in positions]
        print(processed[0].shape)

        processed = np.array(processed, dtype=float)
        print(processed.shape)

        
        policy_output, value_output, logits = self.call(processed)
        policy_output = np.vsplit(policy_output, policy_output.shape[0])
        value_output = np.vsplit(value_output, value_output.shape[0])
        logits = np.vsplit(logits, logits.shape[0])

        policy_output = list(map(lambda x: np.squeeze(x), policy_output))
        value_output = list(map(lambda x: np.squeeze(x), value_output))
        logits = list(map(lambda x: np.squeeze(x), logits))

        return policy_output, value_output


class ModelInferenceFn(tf.keras.layers.Layer):
    def __init__(self, params):
        super(ModelInferenceFn, self).__init__()
        self.params = params
        if FLAGS.bool_features:
            features = tf.dtypes.cast(features, dtype=tf.float32)

        if FLAGS.input_layout == 'nhwc':
            bn_axis = -1
            data_format = 'channels_last'
        else:
            bn_axis = 1
            data_format = 'channels_first'

        self.mg_batchn1 = tf.keras.layers.BatchNormalization(
                            axis=bn_axis,
                            momentum=0.95,
                            epsilon=1e-5,
                            center=True,
                            scale=True,
                            beta_initializer="zeros",
                            gamma_initializer="ones",
                            moving_mean_initializer="zeros",
                            moving_variance_initializer="ones",
                            beta_regularizer=None,
                            gamma_regularizer=None,
                            beta_constraint=None,
                            gamma_constraint=None,
                        )

        self.mg_batchn2 = tf.keras.layers.BatchNormalization(
                            axis=bn_axis,
                            momentum=0.95,
                            epsilon=1e-5,
                            center=True,
                            scale=True,
                            beta_initializer="zeros",
                            gamma_initializer="ones",
                            moving_mean_initializer="zeros",
                            moving_variance_initializer="ones",
                            beta_regularizer=None,
                            gamma_regularizer=None,
                            beta_constraint=None,
                            gamma_constraint=None,
                        )

        self.mg_batchn3 = tf.keras.layers.BatchNormalization(
                            axis=bn_axis,
                            momentum=0.95,
                            epsilon=1e-5,
                            center=True,
                            scale=True,
                            beta_initializer="zeros",
                            gamma_initializer="ones",
                            moving_mean_initializer="zeros",
                            moving_variance_initializer="ones",
                            beta_regularizer=None,
                            gamma_regularizer=None,
                            beta_constraint=None,
                            gamma_constraint=None,
                        )

        self.mg_conv2d1 = tf.keras.layers.Conv2D(
                            filters=params['conv_width'], kernel_size=3, strides=(1, 1), padding='same',
                            data_format=None, dilation_rate=(1, 1), groups=1, activation=None,
                            use_bias=False, kernel_initializer='glorot_uniform',
                            bias_initializer='zeros', kernel_regularizer=None,
                            bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                            bias_constraint=None,
                        )
        self.mg_conv2d2 = tf.keras.layers.Conv2D(
                    filters=params['policy_conv_width'], kernel_size=3, strides=(1, 1), padding='same',
                    data_format=None, dilation_rate=(1, 1), groups=1, activation=None,
                    use_bias=False, kernel_initializer='glorot_uniform',
                    bias_initializer='zeros', kernel_regularizer=None,
                    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                    bias_constraint=None,
                )
        self.mg_conv2d3 = tf.keras.layers.Conv2D(
                    filters=params['value_conv_width'], kernel_size=3, strides=(1, 1), padding='same',
                    data_format=None, dilation_rate=(1, 1), groups=1, activation=None,
                    use_bias=False, kernel_initializer='glorot_uniform',
                    bias_initializer='zeros', kernel_regularizer=None,
                    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                    bias_constraint=None,
                )

        self.mg_global_avgpool2d = tf.keras.layers.AveragePooling2D(pool_size=go.N, 
                                    strides=1, padding='valid', data_format=data_format)

        self.policy_head_dense = tf.keras.layers.Dense(go.N * go.N + 1)
        self.value_head_dense1 = tf.keras.layers.Dense(self.params['fc_width'])
        self.value_head_dense2 = tf.keras.layers.Dense(1)


    def call(self, features, training=True):
        """Builds just the inference part of the model graph.
        Args:
            features: input features tensor.
            training: True if the model is training.
            params: A dictionary
        Returns:
            (policy_output, value_output, logits) tuple of tensors.
        """
        initial_block = self.mg_activation(self.mg_batchn1(self.mg_conv2d1(features)))
        print(initial_block.shape, '!!!!!!!!')
        # the shared stack
        shared_output = initial_block
        for _ in range(self.params['trunk_layers']):
            if FLAGS.use_SE or FLAGS.use_SE_bias:
                shared_output = self.mg_squeeze_excitation_layer(shared_output)
            else:
                shared_output = self.mg_res_layer(shared_output)

        # Policy head
        policy_conv = self.mg_conv2d2(shared_output)
        policy_conv = self.mg_activation(self.mg_batchn2(policy_conv))
        logits = self.policy_head_dense(
            tf.reshape(
                policy_conv, [-1, self.params['policy_conv_width'] * go.N * go.N]))

        policy_output = tf.nn.softmax(logits, name='policy_output')

        # Value head
        value_conv = self.mg_conv2d3(
            shared_output)
        value_conv = self.mg_activation(
            self.mg_batchn3(value_conv))

        value_fc_hidden = self.mg_activation(self.value_head_dense1(
            tf.reshape(value_conv, [-1, self.params['value_conv_width'] * go.N * go.N])))
        value_output = tf.nn.tanh(
            tf.reshape(self.value_head_dense2(value_fc_hidden), [-1]),
            name='value_output')

        return policy_output, value_output, logits
        

    def mg_activation(self, inputs):
        if FLAGS.use_swish:
            return tf.nn.swish(inputs)

        return tf.nn.relu(inputs)

    def residual_inner(self, inputs):
        print(inputs.shape, '?????')
        conv_layer1 = self.mg_batchn1(self.mg_conv2d1(inputs))
        initial_output = self.mg_activation(conv_layer1)
        conv_layer2 = self.mg_batchn1(self.mg_conv2d1(initial_output))
        return conv_layer2

    def mg_res_layer(self, inputs):
        residual = self.residual_inner(inputs)
        output = self.mg_activation(inputs + residual)
        return output

    def mg_squeeze_excitation_layer(self, inputs):
        # Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks.
        # 2018 IEEE/CVF Conference on Computer Vision, 7132-7141.
        # arXiv:1709.01507 [cs.CV]

        channels = self.params['conv_width']
        ratio = FLAGS.SE_ratio
        assert channels % ratio == 0

        residual = self.residual_inner(inputs)
        pool = self.mg_global_avgpool2d(residual)
        fc1 = tf.layers.dense(pool, units=channels // ratio)
        squeeze = self.mg_activation(fc1)

        if FLAGS.use_SE_bias:
            fc2 = tf.layers.dense(squeeze, units=2*channels)
            # Channels_last so axis = 3 = -1
            gamma, bias = tf.split(fc2, 2, axis=3)
        else:
            gamma = tf.layers.dense(squeeze, units=channels)
            bias = 0

        sig = tf.nn.sigmoid(gamma)
        # Explicitly signal the broadcast.
        scale = tf.reshape(sig, [-1, 1, 1, channels])

        excitation = tf.multiply(scale, residual) + bias
        return self.mg_activation(inputs + excitation)


def main(argv):
    """Entry point for running one selfplay game."""
    del argv  # Unused
    params = FLAGS.flag_values_dict()
    model = DualNetwork(save_file=None, params=params)
    strategy = tf.distribute.OneDeviceStrategy(device="/gPU:0")
    with strategy.scope():
        temp = model(np.ones((5, go.N,go.N, 17)))
        print(go.N)
        print(temp[0].shape)
        print(temp[1].shape)
        print(temp[2].shape)
        
    # print(t(np.ones((32,go.N,go.N, 256))))


if __name__ == '__main__':
    app.run(main)