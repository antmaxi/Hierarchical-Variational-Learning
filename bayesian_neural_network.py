# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Trains a Bayesian neural network to classify MNIST digits.

The architecture is LeNet-5 [1].

#### References

[1]: Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner.
     Gradient-based learning applied to document recognition.
     _Proceedings of the IEEE_, 1998.
     http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

As a skeleton used
https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/bayesian_neural_network.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

# Dependency imports
from absl import app
from absl import flags
import matplotlib

from datetime import datetime
# from time import time
# from tensorboard.python.keras.callbacks import TensorBoard

from matplotlib import figure  # pylint: disable=g-import-not-at-top
from matplotlib.backends import backend_agg
import numpy as np
import numpy.matlib
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

import tensorflow as tf
import tensorflow_probability as tfp


from utils import gpu_session

matplotlib.use('Agg')

tf.enable_v2_behavior()

warnings.simplefilter(action='ignore')

try:
    import seaborn as sns  # pylint: disable=g-import-not-at-top

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

tfk = tf.keras
tfkl = tf.keras.layers
tfpd = tfp.distributions
tfpl = tfp.layers

model_type = "dense_layer"
IMAGE_SHAPE = [28, 28, 1]
NUM_TRAIN_EXAMPLES = 60#00  # 000 # 60000
NUM_HELDOUT_EXAMPLES = 10#00  # 000  # 10000
NUM_CLASSES = 10
NUM_GROUPS = 3
# Distribution of digits to groups
LABELS_CHANGE_DICT_GROUPED = {0: 0, 3: 0, 6: 0, 8: 0,  # 0
                              2: 1, 5: 1,  # 1
                              1: 2, 4: 2, 7: 2, 9: 2}  # 2
LABELS_CHANGE_GROUPED = []
for i in range(NUM_CLASSES):
    LABELS_CHANGE_GROUPED.append(LABELS_CHANGE_DICT_GROUPED[i])
LABELS_CHANGE_GROUPED = tuple(LABELS_CHANGE_GROUPED)

flags.DEFINE_float('learning_rate',
                   default=0.001,
                   help='Initial learning rate.')
flags.DEFINE_integer('num_epochs',
                     default=10,
                     help='Number of training steps to run.')
flags.DEFINE_integer('num_grouped_epochs',
                     default=10,
                     help='Number of training steps to run grouped inference.')
flags.DEFINE_integer('batch_size',
                     default=128,
                     help='Batch size.')
flags.DEFINE_string('data_dir',
                    default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                                         'bayesian_neural_network/data'),
                    help='Directory where data is stored (if using real data).')
flags.DEFINE_string(
    'model_dir',
    default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                         'bayesian_neural_network/'),
    help="Directory to put the model's fit.")
integer = flags.DEFINE_integer('viz_steps', default=400, help='Frequency at which save visualizations.')
flags.DEFINE_integer('num_monte_carlo',
                     default=50,
                     help='Network draws to compute predictive probabilities.')
flags.DEFINE_bool('fake_data',
                  default=False,
                  help='If true, uses fake data. Defaults to real data.')

FLAGS = flags.FLAGS


class W0_weights1(object):
    def __init__(self, weights=None):
        self.weights = weights

    def weights_init(self, shape, dtype=None):
        return self.weights  # tfk.backend.random_normal(shape, dtype=dtype)

def get_constant_kernel_prior_fn(loc=0, scale=1.0):
    return tfp.layers.default_mean_field_normal_fn(loc_initializer=tf.constant_initializer(loc),
                                                   untransformed_scale_initializer=tf.constant_initializer(
                                                       tfp.math.softplus_inverse(scale).numpy()))

class DenseVariational(tf.keras.layers.Layer):
  """Dense layer with random `kernel` and `bias`.
  This layer uses variational inference to fit a "surrogate" posterior to the
  distribution over both the `kernel` matrix and the `bias` terms which are
  otherwise used in a manner similar to `tf.keras.layers.Dense`.
  This layer fits the "weights posterior" according to the following generative
  process:
  ```none
  [K, b] ~ Prior()
  M = matmul(X, K) + b
  Y ~ Likelihood(M)
  ```
  """

  def __init__(self,
               units,
               make_posterior_fn,
               make_prior_fn,
               kl_weight=None,
               kl_use_exact=False,
               activation=None,
               use_bias=True,
               activity_regularizer=None,
               **kwargs):
    """Creates the `DenseVariational` layer.
    Arguments:
      units: Positive integer, dimensionality of the output space.
      make_posterior_fn: Python callable taking `tf.size(kernel)`,
        `tf.size(bias)`, `dtype` and returns another callable which takes an
        input and produces a `tfd.Distribution` instance.
      make_prior_fn: Python callable taking `tf.size(kernel)`, `tf.size(bias)`,
        `dtype` and returns another callable which takes an input and produces a
        `tfd.Distribution` instance.
      kl_weight: Amount by which to scale the KL divergence loss between prior
        and posterior.
      kl_use_exact: Python `bool` indicating that the analytical KL divergence
        should be used rather than a Monte Carlo approximation.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation")..
      **kwargs: Extra arguments forwarded to `tf.keras.layers.Layer`.
    """
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)
    super(DenseVariational, self).__init__(
        activity_regularizer=tf.keras.regularizers.get(activity_regularizer),
        **kwargs)
    self.units = int(units)

    self._make_posterior_fn = make_posterior_fn
    self._make_prior_fn = make_prior_fn
    self._kl_divergence_fn = _make_kl_divergence_penalty(
        kl_use_exact, weight=kl_weight)

    self.activation = tf.keras.activations.get(activation)
    self.use_bias = use_bias
    self.supports_masking = False
    self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)

  def build(self, input_shape):
    dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `Dense` layer with non-floating point '
                      'dtype %s' % (dtype,))
    input_shape = tf.TensorShape(input_shape)
    last_dim = tf.compat.dimension_value(input_shape[-1])
    if last_dim is None:
      raise ValueError('The last dimension of the inputs to `DenseVariational` '
                       'should be defined. Found `None`.')
    self.input_spec = tf.keras.layers.InputSpec(
        min_ndim=2, axes={-1: last_dim})

    self._posterior = self._make_posterior_fn(
        last_dim * self.units,
        self.units if self.use_bias else 0,
        dtype)
    self._prior = self._make_prior_fn(
        last_dim * self.units,
        self.units if self.use_bias else 0,
        dtype)

    self.built = True

  def call(self, inputs):
    dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
    inputs = tf.cast(inputs, dtype, name='inputs')

    q = self._posterior(inputs)
    r = self._prior(inputs)
    self.add_loss(self._kl_divergence_fn(q, r))

    w = tf.convert_to_tensor(value=q)
    prev_units = self.input_spec.axes[-1]
    if self.use_bias:
      split_sizes = [prev_units * self.units, self.units]
      kernel, bias = tf.split(w, split_sizes, axis=-1)
    else:
      kernel, bias = w, None

    kernel = tf.reshape(kernel, shape=tf.concat([
        tf.shape(kernel)[:-1],
        [prev_units, self.units],
    ], axis=0))
    outputs = tf.matmul(inputs, kernel)

    if self.use_bias:
      outputs = tf.nn.bias_add(outputs, bias)

    if self.activation is not None:
      outputs = self.activation(outputs)  # pylint: disable=not-callable

    return outputs


def _make_kl_divergence_penalty(
    use_exact_kl=False,
    test_points_reduce_axis=(),  # `None` == "all"; () == "none".
    test_points_fn=tf.convert_to_tensor,
    weight=None):
  """Creates a callable computing `KL[a,b]` from `a`, a `tfd.Distribution`."""

  if use_exact_kl:
    kl_divergence_fn = tfpd.kullback_leibler.kl_divergence
  else:
    def kl_divergence_fn(distribution_a, distribution_b):
      z = test_points_fn(distribution_a)
      return tf.reduce_mean(
          distribution_a.log_prob(z) - distribution_b.log_prob(z),
          axis=test_points_reduce_axis)

  # Closure over: kl_divergence_fn, weight.
  def _fn(distribution_a, distribution_b):
    """Closure that computes KLDiv as a function of `a` as in `KL[a, b]`."""
    with tf.name_scope('kldivergence_loss'):
      kl = kl_divergence_fn(distribution_a, distribution_b)
      if weight is not None:
        kl = tf.cast(weight, dtype=kl.dtype) * kl
      # Losses appended with the model.add_loss and are expected to be a single
      # scalar, unlike model.loss, which is expected to be the loss per sample.
      # Therefore, we reduce over all dimensions, regardless of the shape.
      # We take the sum because (apparently) Keras will add this to the *post*
      # `reduce_sum` (total) loss.
      # TODO(b/126259176): Add end-to-end Keras/TFP test to ensure the API's
      # align, particularly wrt how losses are aggregated (across batch
      # members).
      return tf.reduce_sum(kl, name='batch_total_kl_divergence')

  return _fn

def create_model(model_type="LeNet", input_size=(NUM_TRAIN_EXAMPLES,), output_size=NUM_CLASSES):
    """Creates a Keras model using the LeNet-5 architecture.

  Returns:
      model: Compiled Keras model.
  """
    # KL divergence weighted by the number of training samples, using
    # lambda function to pass as input to the kernel_divergence_fn on
    # flipout layers.
    # TODO: check correctness for per-group model

    # Define a LeNet-5 model using three convolutional (with max pooling)
    # and two fully connected dense layers. We use the Flipout
    # Monte Carlo estimator for these layers, which enables lower variance
    # stochastic gradients than naive reparameterization.

    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    # We use the categorical_crossentropy loss since the MNIST dataset contains
    # ten labels. The Keras API will then automatically add the
    # Kullback-Leibler divergence (contained on the individual layers of
    # the model), to the cross entropy loss, effectively
    # calcuating the (negated) Evidence Lower Bound Loss (ELBO)

    if model_type == "LeNet":
        kl_divergence_function = (lambda q, p, _: tfpd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                                                  tf.cast(input_size, dtype=tf.float32))
        model = tf.keras.models.Sequential([
            tfp.layers.Convolution2DFlipout(
                6, kernel_size=5, padding='SAME',
                kernel_divergence_fn=kl_divergence_function,
                activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(
                pool_size=[2, 2], strides=[2, 2],
                padding='SAME'),
            tfp.layers.Convolution2DFlipout(
                16, kernel_size=5, padding='SAME',
                kernel_divergence_fn=kl_divergence_function,
                activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(
                pool_size=[2, 2], strides=[2, 2],
                padding='SAME'),
            tfp.layers.Convolution2DFlipout(
                120, kernel_size=5, padding='SAME',
                kernel_divergence_fn=kl_divergence_function,
                activation=tf.nn.relu),
            tf.keras.layers.Flatten(),
            tfp.layers.DenseFlipout(
                84, kernel_divergence_fn=kl_divergence_function,
                activation=tf.nn.relu),
            tfp.layers.DenseFlipout(
                output_size, kernel_divergence_fn=kl_divergence_function,
                activation=tf.nn.softmax)
        ])
        model.compile(optimizer, loss='categorical_crossentropy',
                      metrics=['accuracy'], experimental_run_tf_function=False)
        return model
    elif model_type == "dense_layer":
        tau_0_inv = 1 / 1000.0
        v = 100.0
        models = []
        input = tfk.Input(shape=tuple(IMAGE_SHAPE), name='img')
        if 0:
            for i in range(NUM_GROUPS):
                l1 = tfp.layers.Convolution2DFlipout(
                    6, kernel_size=5, padding='SAME',
                    kernel_divergence_fn=kl_divergence_function,
                    activation=tf.nn.relu)(input)
                x1 = l1(inputs)
                l2 = tf.keras.layers.MaxPooling2D(
                    pool_size=[2, 2], strides=[2, 2],
                    padding='SAME')(l1)
                l3 = tfp.layers.Convolution2DFlipout(
                    16, kernel_size=5, padding='SAME',
                    kernel_divergence_fn=kl_divergence_function,
                    activation=tf.nn.relu)(l2)
                l4 = tf.keras.layers.MaxPooling2D(
                    pool_size=[2, 2], strides=[2, 2],
                    padding='SAME')(l3)
                l5 = tfp.layers.Convolution2DFlipout(
                    120, kernel_size=5, padding='SAME',
                    kernel_divergence_fn=kl_divergence_function,
                    activation=tf.nn.relu)(l4)
                l6 = tf.keras.layers.Flatten()(l5)
                l7 = tfp.layers.DenseFlipout(
                    84, kernel_divergence_fn=kl_divergence_function,
                    activation=tf.nn.relu)(l6)
                l8 = tfp.layers.DenseFlipout(
                    output_size, kernel_divergence_fn=kl_divergence_function,
                    activation=tf.nn.softmax)(l7)
                models.append(tfk.Model(inputs=inputs, outputs=l8, name='dense_net_{}'.format(i)))
                models[i].compile(optimizer, loss='categorical_crossentropy',
                                  metrics=['accuracy'], experimental_run_tf_function=False)
        if 1:
            #TODO: batch size? Need to weigh with 1/#minibatches
            kl_divergence_function = (lambda q, p, _: tfpd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                                                      tf.cast(input_size, dtype=tf.float32))
            f = tfkl.Flatten()(input)
            #f_big =
            posterior_0_distribution = tfpl.util.default_mean_field_normal_fn(#TODO: right?
                loc_initializer=tf.zeros_initializer,
                untransformed_scale_initializer=tf.keras.initializers.constant(
                    tfp.bijectors.Softplus().inverse(tau_0_inv).numpy()))

                #tf.constant_initializer(tfp.math.softplus_inverse(tau_0_inv).numpy()))
            W0 = tfpl.DenseFlipout(
                output_size, kernel_divergence_fn=kl_divergence_function,
                kernel_posterior_fn=posterior_0_distribution,
                bias_posterior_fn=posterior_0_distribution,
                activation=tf.nn.softmax)
            w0 = W0(f)
            #a = W0.__call__
            W0_weights = W0.get_weights()
            tr = W0.trainable_weights
            # Get group weights variances
            tau_g_inv = tf.square(tfp.edward2.HalfNormal(scale=v * tf.ones([NUM_GROUPS])))

            # Per group networks
            models = []
            denses = []
            #w0 = W0_weights1(weights=x.get_weights()[0])
            # b = w0.weights
            inputs = []
            flattens = []
            outputs = []
            for i in range(NUM_GROUPS):
                inputs.append(tfk.Input(shape=tuple(IMAGE_SHAPE), name='img{}'.format(i)))
                flattens.append(tfkl.Flatten()(inputs[i]))
                kl_divergence_function_g = (lambda q, p, _: tfpd.kl_divergence(q, p) /
                                                            tf.cast(input_size[i], dtype=tf.float32))
                posterior_g_distribution = tfpl.util.default_mean_field_normal_fn(
                    loc_initializer=tf.constant_initializer(W0.get_weights()[1]),
                    untransformed_scale_initializer=tf.square(tfp.edward2.HalfNormal(scale=v)))
                denses.append(tfpl.DenseFlipout(
                    output_size,
                    kernel_divergence_fn=kl_divergence_function_g,
                    kernel_posterior_fn=posterior_g_distribution,
                    bias_prior_fn=posterior_g_distribution,
                    activation=tf.nn.softmax))
                outputs.append(denses[i](flattens[i]))
                models.append(tfk.Model(inputs=inputs[i], outputs=outputs[i], name=f'dense_net_{i}'))
            model = tfk.Model(inputs=inputs, outputs=outputs)
            for i in range(NUM_GROUPS):
                model.add_loss(models[i].losses)
            model.compile(optimizer, #loss=y.losses, #['categorical_crossentropy'] * NUM_GROUPS,
                          metrics=['accuracy'], experimental_run_tf_function=False)
            # tfk.utils.plot_model(model, to_file='multi_input_and_output_model.png', show_shapes=True)

            for i in range(NUM_GROUPS):
                models[i].compile(optimizer, loss='categorical_crossentropy',
                                  metrics=['accuracy'], experimental_run_tf_function=False)
        return models, model, w


class MNISTSequence(tf.keras.utils.Sequence):
    """Produces a sequence of MNIST digits with labels."""

    def __init__(self, data=None, batch_size=128, used_labels=tuple(range(10)), labels_len=NUM_CLASSES,
                 labels_change=tuple(range(10)), preprocessing=True, fake_data_size=None):
        """Initializes the sequence.

    Args:
      data: Tuple of numpy `array` instances, the first representing images and
            the second labels.
      batch_size: Integer, number of elements in each training batch.
      fake_data_size: Optional integer number of fake datapoints to generate.
    """
        if data:
            images, labels = data
        else:
            images, labels = MNISTSequence.__generate_fake_data(
                num_images=fake_data_size, num_classes=NUM_CLASSES)
        if preprocessing:
            self.images, self.labels = MNISTSequence.__preprocessing(
                images, labels, used_labels, labels_change, labels_len)
        else:
            self.images, self.labels = images, labels
        self.batch_size = batch_size

    @staticmethod
    def __generate_fake_data(num_images, num_classes):
        """Generates fake data in the shape of the MNIST dataset for unittest.

    Args:
      num_images: Integer, the number of fake images to be generated.
      num_classes: Integer, the number of classes to be generate.
    Returns:
      images: Numpy `array` representing the fake image data. The
              shape of the array will be (num_images, 28, 28).
      labels: Numpy `array` of integers, where each entry will be
              assigned a unique integer.
    """
        images = np.random.randint(low=0, high=256,
                                   size=(num_images, IMAGE_SHAPE[0],
                                         IMAGE_SHAPE[1]))
        labels = np.random.randint(low=0, high=num_classes,
                                   size=num_images)
        return images, labels

    @staticmethod
    def __preprocessing(images, labels, used_labels, labels_change, labels_len):
        """Preprocesses image and labels data.

    Args:
      images: Numpy `array` representing the image data.
      labels: Numpy `array` representing the labels data (range 0-9).

    Returns:
      images: Numpy `array` representing the image data, normalized
              and expanded for convolutional network input.
      labels: Numpy `array` representing the labels data (range 0-9),
              as one-hot (categorical) values.
    """
        # Auxiliary dicts for integer label - its MNIST representation mapping
        # labels_bin_dict = {i: tuple(np.identity(10)[i]) for i in range(10)}
        # bin_labels_dict = {v: k for k, v in labels_bin_dict.items()}

        # Get indices of used labels using convertation of MNIST labels from array to integer system
        indices = []
        for label in labels:
            indices.append(int(label) in used_labels)
        # Select used labels
        labels_transformed = []
        for index, flag in enumerate(indices):
            if flag:
                labels_transformed.append(labels_change[labels[index]])
        # Normalize images
        images = 2 * (images[indices] / 255.) - 1.
        images = images[..., tf.newaxis]
        # Convert labels for cross-entropy loss
        labels = tf.keras.utils.to_categorical(y=labels_transformed, num_classes=labels_len)

        return images, labels

    def __len__(self):
        return int(tf.math.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


def train_model(model, train_seq, heldout_seq=((), ()), tensorboard_callback=None):
    """
    Trains LeNet model on MNIST data in a flexible to data way

    :param model:
    :param train_seq:
    :param heldout_seq:
    :param tensorboard_callback:
    :return: trained model
    """

    print(' ... Training neural network')
    for epoch in range(FLAGS.num_epochs):
        print('Epoch {}'.format(epoch))
        epoch_accuracy, epoch_loss = [], []
        for step, (batch_x, batch_y) in enumerate(train_seq):
            batch_loss, batch_accuracy = model.train_on_batch(
                batch_x, batch_y)
            epoch_accuracy.append(batch_accuracy)
            epoch_loss.append(batch_loss)

            if step % 100 == 0:
                print('Epoch: {}, Batch index: {}, '
                      'Loss: {:.3f}, Accuracy: {:.3f}'.format(
                    epoch, step,
                    tf.reduce_mean(epoch_loss),
                    tf.reduce_mean(epoch_accuracy)))
    if 0:
        training_history = model1.fit(
            train_seq.images,  # input
            train_seq.labels,  # output
            batch_size=FLAGS.batch_size,
            verbose=1,  # Suppress chatty output; use Tensorboard instead
            epochs=FLAGS.num_epochs,
            # validation_data=(x_test, y_test),
            callbacks=[tensorboard_callback],
        )
    return model


def get_labels_groups(labels_change_dict_grouped):
    """
    :param labels_change_dict_grouped: dict {class: group}
    :return: NUM_GROUPS-length tuple of tuples of corresp. labels
    """
    res = []
    for val in range(NUM_GROUPS):
        res_aux = []
        for key, value in labels_change_dict_grouped.items():
            if value == val:
                res_aux.append(key)
        res.append(tuple(res_aux))
    return tuple(res)


def main(argv):
    logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    train_set, heldout_set = tf.keras.datasets.mnist.load_data(path='mnist.npz')

    # Cut sets if needed for optimization
    train_set = (train_set[0][0:NUM_TRAIN_EXAMPLES], train_set[1][0:NUM_TRAIN_EXAMPLES])
    heldout_set = (heldout_set[0][0:NUM_HELDOUT_EXAMPLES], heldout_set[1][0:NUM_HELDOUT_EXAMPLES])

    # NUM_GROUPS-length tuple of tuples of corresp. labels, e.g. ((0,3,6,8), (2,5), (1,4,7,9))
    labels_groups = get_labels_groups(LABELS_CHANGE_DICT_GROUPED)
    # Training grouped model (for prediciton) on the whole data but with merged classes
    train_seq_grouped = MNISTSequence(data=train_set, batch_size=FLAGS.batch_size,
                                      labels_change=LABELS_CHANGE_GROUPED, labels_len=NUM_GROUPS)
    model_grouped = create_model(model_type="LeNet",
                                 input_size=np.shape(train_seq_grouped.labels)[0],
                                 output_size=NUM_GROUPS)
    model_grouped = train_model(model_grouped, train_seq_grouped, )

    # Creating hierarchical model
    train_seqs = []
    train_images = []
    train_labels = []
    for i in range(NUM_GROUPS):
        train_seqs.append(MNISTSequence(data=train_set, batch_size=FLAGS.batch_size,
                                        used_labels=labels_groups[i],
                                        labels_len=NUM_CLASSES,
                                        labels_change=tuple(range(10)), ))
        train_images.append(train_seqs[i].images)
        train_labels.append(train_seqs[i].labels)
    models, model, w_0 = create_model(model_type="dense_layer",
                                    input_size=[np.shape(train_seq_now.labels)[0] for train_seq_now in train_seqs],
                                    output_size=NUM_CLASSES)
    #print(w_0.get_weights()[0][0])

    # Training the hierarchical model
    print(' ... Training neural network')
    for i in range(NUM_GROUPS):
        models[i].fit(train_images[i], train_labels[i],
                  batch_size=FLAGS.batch_size,
                  verbose=1,  # Suppress chatty output; use Tensorboard instead
                  epochs=FLAGS.num_epochs,
                  # validation_data=(x_test, y_test),
                  callbacks=[tensorboard_callback], )
    if 0:
        for epoch in range(FLAGS.num_epochs):
            print('Epoch {}'.format(epoch))
            epoch_accuracy, epoch_loss = [], []
            for i in np.random.permutation(tuple(range(NUM_GROUPS))):
                # print(w_0.get_weights()[0][0,0])
                # print(w.numpy()[0][0])
                for step, (batch_x, batch_y) in enumerate(train_seqs[i]):
                    batch_loss, batch_accuracy = models[i].train_on_batch(
                        batch_x, batch_y)
                    epoch_accuracy.append(batch_accuracy)
                    epoch_loss.append(batch_loss)
                    # print(models[0].get_weights()[0][0][0])
                    # print(models[1].get_weights()[0][0][0])
                    # print(models[2].get_weights()[0][0][0])
                    # print(w.numpy()[0][0])

                    if step % 100 == 0:
                        print('Epoch: {}, Batch index: {}, '
                              'Loss: {:.3f}, Accuracy: {:.3f}'.format(
                            epoch, step,
                            tf.reduce_mean(epoch_loss),
                            tf.reduce_mean(epoch_accuracy)))
    # Test
    heldout_seq_grouped = MNISTSequence(data=heldout_set, batch_size=FLAGS.batch_size,
                                        labels_change=LABELS_CHANGE_GROUPED, labels_len=NUM_GROUPS)
    heldout_seq_classes = MNISTSequence(data=heldout_set, batch_size=FLAGS.batch_size, labels_len=NUM_CLASSES)

    # Predict groups
    predicted_groups_probs = model_grouped.predict(x=heldout_seq_grouped.images, batch_size=None, verbose=1)
    print("Got predicted groups")

    # Depending on group predict class
    predictions_right = []
    # for i in range(np.shape(heldout_seq_grouped.labels)[0]):
    # Get corresponding to group data
    res = np.zeros((np.shape(heldout_seq_grouped.labels)[0], NUM_CLASSES))
    for j in range(NUM_GROUPS):
        predicted = models[j].predict(x=heldout_seq_classes.images, batch_size=None, verbose=1)
        a = np.matlib.repmat(predicted_groups_probs[:, j], NUM_CLASSES, 1)
        res += np.multiply(a.transpose(), predicted)
    labels_bin_dict = np.identity(10)
    for i in range(np.shape(heldout_seq_grouped.labels)[0]):
        pos = np.where(res[i, :] == np.amax(res[i, :]))[0]
        print("{}-{}".format(labels_bin_dict[pos][0], heldout_seq_classes.labels[i]))
        predictions_right.append(np.array_equal(
            labels_bin_dict[pos][0], heldout_seq_classes.labels[i])
        )
    true_all = np.count_nonzero(predictions_right)
    number_all = len(predictions_right)
    print('Final results: {}/{} = {}'.format(true_all, number_all, true_all / number_all))


if __name__ == '__main__':
    #gpu_session(num_gpus=1)
    app.run(main)
