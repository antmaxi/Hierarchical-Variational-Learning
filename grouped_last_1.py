from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import math
import os
from utils import gpu_session
from absl import app

from pathlib import Path
from datetime import datetime
import json
import argparse

import tensorflow as tf
import tensorflow_probability as tfp

tf.compat.v1.disable_eager_execution()

# log_dir = "/tmp/tfdbg2_logdir"
# tf.debugging.experimental.enable_dump_debug_info(log_dir, tensor_debug_mode="FULL_TENSOR", circular_buffer_size=100)

tfk = tf.keras
tfkl = tf.keras.layers
tfkb = tf.keras.backend
tfpd = tfp.distributions
tfpl = tfp.layers

model_type = "dense_layer"
IMAGE_SHAPE = [28, 28, 1]
NUM_TRAIN_EXAMPLES = 60000
NUM_HELDOUT_EXAMPLES = 10000
NUM_CLASSES = 10
NUM_GROUPS = 3

gpu = 0
if gpu:
    LOG_DIR = '/local/home/antonma/my_tf_logs_'
else:
    LOG_DIR = '/home/anton/my_tf_logs_'
LOG_DIR = LOG_DIR + datetime.now().strftime("%Y%m%d-%H%M%S")

LABELS_CHANGE_DICT_GROUPED = {0: 0, 3: 0, 6: 0, 8: 0,  # 0
                              2: 1, 5: 1,  # 1
                              1: 2, 4: 2, 7: 2, 9: 2}  # 2

LABELS_CHANGE_GROUPED = []  # (0, 2, 1, 0, ...)
for i in range(NUM_CLASSES):
    LABELS_CHANGE_GROUPED.append(LABELS_CHANGE_DICT_GROUPED[i])
LABELS_CHANGE_GROUPED = tuple(LABELS_CHANGE_GROUPED)

# build a lookup table
TABLE = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(np.array(list(LABELS_CHANGE_DICT_GROUPED.keys())), dtype=np.int32),
        values=tf.constant(np.array(list(LABELS_CHANGE_DICT_GROUPED.values())), dtype=np.int32),
    ),
    default_value=tf.constant(-1),
    name="class_to_group"
)

parser = argparse.ArgumentParser(description='Choose the type of execution.')
parser.add_argument("--config_path",
                    type=Path,
                    help="Path to the main json config. "
                         "Ex: 'configurations/femnist_virtual.json'",
                    default='configurations/mnist_virtual.json')
parser.add_argument('--lr', help='Initial learning rate.', type=float,
                    default=0.001)
parser.add_argument('--num_epochs', help='Number of training steps to run.', type=int,
                    default=100)
parser.add_argument('--num_epochs_g', help='Number of training steps to run grouped inference.', type=int,
                    default=3)
parser.add_argument('--num_sample', help='Number of Monte-Carlo sampling repeats.', type=int,
                    default=50)
parser.add_argument('--batch', help='Batch size.', type=int,
                    default=128)

parser.add_argument('--gpu', help="To run on GPU or not", type=int,
                    default=0)

parser.add_argument('--data_dir', help='Directory where data is stored (if using real data).',
                    default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                                         'bayesian_neural_network/data'))
parser.add_argument('--model_dir', help="Directory to put the model's fit.",
                    default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                                         'bayesian_neural_network/data'))
args = parser.parse_args()


def soft_inv(x):
    return np.log(np.exp(x) - 1)


class MNISTSequence(tf.keras.utils.Sequence):
    """Produces a sequence of MNIST digits with labels."""

    def __init__(self, images=None, labels=None, labels_to_binary=True, batch_size=args.batch,
                 used_labels=tuple(range(10)), labels_len=NUM_CLASSES,
                 labels_change=tuple(range(10)), preprocessing=True, fake_data_size=None):
        """Initializes the sequence.

    Args:
      data: Tuple of numpy `array` instances, the first representing images and
            the second labels.
      batch_size: Integer, number of elements in each training batch.
      fake_data_size: Optional integer number of fake datapoints to generate.
    """
        if images is None:
            images, labels = MNISTSequence.__generate_fake_data(
                num_images=fake_data_size, num_classes=NUM_CLASSES)
        if preprocessing:
            self.images, self.labels = MNISTSequence.__preprocessing(
                images, labels, used_labels, labels_change, labels_len)
        else:
            images = 2 * (images / 255.) - 1.
            if labels_to_binary:
                labels = tf.keras.utils.to_categorical(labels)

            self.images = np.array([np.append(images[i].flatten(), labels[i])
                                    for i in range(np.shape(images)[0])])
            self.labels = labels
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
        return int(math.ceil(np.shape(self.images)[0] / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


def create_model(type="dens1", output_size=NUM_CLASSES):
    """Creates a Keras model using the LeNet-5 architecture.
    type: "dens1" or "lenet"
  Returns:
      model: Compiled Keras model.
  """
    # KL divergence weighted by the number of training samples, using
    # lambda function to pass as input to the kernel_divergence_fn on
    # flipout layers.
    kl_divergence_function = (lambda q, p, _: tfpd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                                              tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))

    # Define a LeNet-5 model using three convolutional (with max pooling)
    # and two fully connected dense layers. We use the Flipout
    # Monte Carlo estimator for these layers, which enables lower variance
    # stochastic gradients than naive reparameterization.
    if type == "lenet":
        model = tf.keras.models.Sequential([
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
    elif type == "dens1":
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tfp.layers.DenseFlipout(
                output_size, kernel_divergence_fn=kl_divergence_function,
                activation=tf.nn.softmax),
        ])
    else:
        return -1
    # Model compilation.
    optimizer = tf.keras.optimizers.Adam(lr=args.lr)
    # We use the categorical_crossentropy loss since the MNIST dataset contains
    # ten labels. The Keras API will then automatically add the
    # Kullback-Leibler divergence (contained on the individual layers of
    # the model), to the cross entropy loss, effectively
    # calcuating the (negated) Evidence Lower Bound Loss (ELBO)
    model.compile(optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'], experimental_run_tf_function=False)
    return model


def train_model(model, train_seq, epochs=args.num_epochs, heldout_seq=((), ()), tensorboard_callback=None):
    """
    Trains LeNet model on MNIST data in a flexible to data way

    :param epochs:
    :param model:
    :param train_seq:
    :param heldout_seq:
    :param tensorboard_callback:
    :return: trained model
    """

    tensorboard = tfk.callbacks.TensorBoard(
        log_dir=LOG_DIR,
        histogram_freq=0,
        batch_size=args.batch,
        write_graph=True,
        write_grads=True
    )
    tensorboard.set_model(model)

    def named_logs(model, logs):
        result = {}
        for l in zip(model.metrics_names, logs):
            result[l[0]] = l[1]
        return result

    if 0:
        for epoch in range(epochs):
            print('Epoch {}'.format(epoch))
            epoch_accuracy, epoch_loss = [], []
            for step, (batch_x, batch_y) in enumerate(train_seq):
                logs = model.train_on_batch(
                    batch_x, batch_y)
                tensorboard.on_epoch_end(step, named_logs(model, logs))

    else:
        training_history = model.fit(
            train_seq.images,  # input
            train_seq.labels,  # output
            batch_size=args.batch,
            verbose=1,
            epochs=epochs,
            # validation_data=(x_test, y_test),
            callbacks=[tensorboard],
        )
    return model


class DenseVariationalGrouped(tfkl.Layer):
    def __init__(self,
                 units,
                 kl_weight,
                 activation=None,
                 tau_inv_0=None,
                 v=None,
                 init_sigma_of_mu=None,
                 init_rho=None,
                 num_groups=NUM_GROUPS,
                 num_sample=args.num_sample,
                 **kwargs):
        self.units = units
        self.kl_weight = kl_weight
        self.activation = tfk.activations.get(activation)
        self.tau_inv_0 = tau_inv_0
        self.v = v
        self.init_sigma_of_mu = init_sigma_of_mu
        self.init_rho = init_rho
        self.num_groups = num_groups
        self.num_sample = num_sample
        # Variational parameters
        self.kernel_mu = None
        self.bias_mu = None
        self.kernel_rho = None
        self.bias_rho = None
        self.kernel_mu_g = None
        self.bias_mu_g = None
        self.kernel_rho_g = None
        self.bias_rho_g = None
        self.tau_g = None
        self.gamma_g = None
        self.gamma_mu = None
        self.gamma_rho = None
        #self.k = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        super().__init__(**kwargs)

    def build(self, input_shape):
        print(input_shape)

        self.kernel_mu = self.add_weight(name='kernel_mu',
                                         shape=(input_shape[1] - NUM_CLASSES, self.units),
                                         initializer=tf.constant_initializer(value=0),
                                         trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu',
                                       shape=(self.units,),
                                       initializer=tf.constant_initializer(value=0),
                                       trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho',
                                          shape=(input_shape[1] - NUM_CLASSES, self.units),
                                          initializer=tf.constant_initializer(value=soft_inv(self.tau_inv_0 / 2.0)),
                                          trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho',
                                        shape=(self.units,),
                                        initializer=tf.constant_initializer(value=soft_inv(self.tau_inv_0 / 2.0)),
                                        trainable=True)
        self.kernel_mu_g = []
        self.bias_mu_g = []
        self.kernel_rho_g = []
        self.bias_rho_g = []
        super().build(input_shape)

    def kl_loss(self, w, mu_var, sigma_var, mu_prior, sigma_prior):
        variational_dist = tfp.distributions.Normal(mu_var, sigma_var)
        return self.kl_weight * tfkb.sum(variational_dist.log_prob(w)
                                         - self.log_prior_prob(w, mu_prior, sigma_prior)
                                         )

    def log_prior_prob(self, w, mu, sigma):
        prior_dist = tfp.distributions.Normal(tf.zeros_like(w) + mu, tf.zeros_like(w) + sigma)
        return prior_dist.log_prob(w)

    def call(self, inputs, training=False, **kwargs):
        eps = 0

        imgs = inputs[:, : -NUM_CLASSES]
        y = inputs[:, -NUM_CLASSES:]

        y_preds = TABLE.lookup(tf.cast(tf.math.argmax(y, axis=1), dtype=tf.int32))  # get groups from classes

        kernel_sigma = tf.math.softplus(self.kernel_rho) + eps
        bias_sigma = tf.math.softplus(self.bias_rho) + eps
        print("???")
        # print(tfkb.get_value(self.bias_rho))
        # print(tfkb.get_value(self.kernel_rho))
        # print(tfkb.get_value(self.gamma_rho))
        # print(tfkb.get_value(bias_sigma))

        n = imgs.get_shape().as_list()[0]
        result = tf.zeros_like(tfkb.dot(imgs, self.kernel_mu))
        # Monte-Carlo for loss
        # TODO: is it possible not to calculate loss when predicting?

        #self.k.assign(0.0)
        for step in range(self.num_sample):
            loss = 0.0
            kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)
            loss += self.kl_loss(kernel, self.kernel_mu, kernel_sigma, 0, self.tau_inv_0)

            bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)
            loss += self.kl_loss(bias, self.bias_mu, bias_sigma, 0, self.tau_inv_0)


            # print(tfkb.get_value(bias_sigma_g))
            # tau_inv_g = tf.ones_like(tau_inv_g)
            # print(tfkb.get_value(tau_inv_g))

            # throw out if loss is None (numerically too small argument of logarithm appeared somewhere)
            if tf.math.is_nan(loss):
                loss = 0.0
            self.add_loss(loss / self.num_sample)
            result += self.activation(tfkb.dot(imgs, kernel) + bias) / self.num_sample
        return result
        #if tf.math.count_nonzero(k):
        #else:
        #    return self.activation(result)


# def neg_log_likelihood(y_obs, y_pred, sigma=1):
#    dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
#    return tfkb.sum(-dist.log_prob(y_obs))


import warnings


def main(argv):
    warnings.filterwarnings('ignore')
    # log_dir = "/local/home/antonma/HFL/tfdbg2_logdir"
    # log_dir = "/tmp/tfdbg2_logdir"
    if not gpu:
        with args.config_path.absolute().open(mode='r') as config_file:
            configs = json.load(config_file)

        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Configs
        data_set_conf = configs['data_set_conf']
        training_conf = configs['training_conf']
        model_conf = configs['model_conf']

        all_params = {**data_set_conf,
                      **training_conf,
                      **model_conf, }

    prior_params = {
        # "init_sigma_of_mu": 1.0,
        # "init_rho": -3,
        "tau_inv_0": 100.0,  # prior sigma of weights
        "v": 10.0,  # prior sigma of gammas
        "num_groups": 3,
    }

    train_set, heldout_set = tf.keras.datasets.mnist.load_data(path='mnist.npz')

    train_seq = MNISTSequence(images=train_set[0], labels=train_set[1], batch_size=args.batch,
                              preprocessing=False)

    inputs = tfk.Input(shape=(IMAGE_SHAPE[0] * IMAGE_SHAPE[1] + NUM_CLASSES,), name='img')
    kl_weight = 1.0 / (NUM_TRAIN_EXAMPLES / float(args.batch))
    output1 = DenseVariationalGrouped(NUM_CLASSES, kl_weight,
                                      activation="softmax",
                                      **prior_params)(inputs)
    #output = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax)(output1)
    model = tfk.Model(inputs=inputs, outputs=output1)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  optimizer=tfk.optimizers.Adam(lr=args.lr),
                  metrics=['accuracy']
                  )
    model.build(input_shape=[None, IMAGE_SHAPE[0] * IMAGE_SHAPE[1] + NUM_CLASSES])
    print(model.summary())

    print(' ... Training main network')
    train_model(model, train_seq, epochs=args.num_epochs)
    # print(model.layers[1].gamma_g)
    # GROUP inference model
    model_grouped = create_model(type="lenet", output_size=NUM_GROUPS)
    train_seq_grouped = MNISTSequence(images=train_set[0], labels=train_set[1], batch_size=args.batch,
                                      labels_change=LABELS_CHANGE_GROUPED, labels_len=NUM_GROUPS,
                                      preprocessing=True)
    print(' ... Training group inference network')
    train_model(model_grouped, train_seq_grouped, epochs=args.num_epochs_g)
    # Predict groups
    heldout_seq_grouped = MNISTSequence(images=heldout_set[0], labels=heldout_set[1], batch_size=args.batch,
                                        labels_change=LABELS_CHANGE_GROUPED, labels_len=NUM_GROUPS,
                                        preprocessing=True)
    print(" ... Predicting groups")
    predicted_groups_probs = model_grouped.predict(x=heldout_seq_grouped.images, batch_size=None, verbose=1)
    predicted_groups_probs = np.hstack((predicted_groups_probs,
                                        np.zeros((np.shape(predicted_groups_probs)[0], NUM_CLASSES - NUM_GROUPS))))
    # images with appended
    test_seq = MNISTSequence(images=heldout_set[0], labels=predicted_groups_probs, labels_to_binary=False,
                             batch_size=args.batch,
                             preprocessing=False)
    result_prob = model.predict(test_seq.images)
    result_argmax = np.argmax(result_prob, axis=1)

    print(sum(1 for x, y in zip(heldout_set[1], result_argmax) if x == y) / float(len(result_argmax)))
    print(result_argmax)
    print(heldout_set[1])
    if 0:
        def neg_log_likelihood(y_obs, y_pred, sigma=noise):
            dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
            return tfkb.sum(-dist.log_prob(y_obs))


if __name__ == '__main__':
    if gpu:
        gpu_session(num_gpus=1)  #
    app.run(main)
