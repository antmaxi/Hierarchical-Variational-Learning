from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import os
import warnings

# Dependency imports
from absl import app
from absl import flags

from datetime import datetime
# from time import time
# from tensorboard.python.keras.callbacks import TensorBoard

from matplotlib import figure  # pylint: disable=g-import-not-at-top
from matplotlib.backends import backend_agg
import numpy.matlib
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

import tensorflow as tf
import tensorflow_probability as tfp

import argparse
import tqdm

tfk = tf.keras
tfkl = tf.keras.layers
tfkb = tf.keras.backend
tfpd = tfp.distributions
tfpl = tfp.layers

model_type = "dense_layer"
IMAGE_SHAPE = [28, 28, 1]
NUM_TRAIN_EXAMPLES = 60  # 00  # 000 # 60000
NUM_HELDOUT_EXAMPLES = 10  # 00  # 000  # 10000
NUM_CLASSES = 10
NUM_GROUPS = 3

import tensorflow as tf
import tensorflow_probability as tfp

parser = argparse.ArgumentParser(description='Choose the type of execution.')
parser.add_argument('--lr', help='Initial learning rate.', type=float,
                    default=0.001)
parser.add_argument('--epochs', help='Number of training steps to run.', type=int,
                    default=2)
parser.add_argument('--epochs_g', help='Number of training steps to run grouped inference.', type=int,
                    default=10)
parser.add_argument('--num_sample', help='Number of Monte-Carlo sampling repeats.', type=int,
                    default=10)
parser.add_argument('--batch', help='Batch size.', type=int,
                    default=128)
parser.add_argument('--data_dir', help='Directory where data is stored (if using real data).',
                    default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                                         'bayesian_neural_network/data'))
parser.add_argument('--model_dir', help="Directory to put the model's fit.",
                    default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                                         'bayesian_neural_network/data'))
args = parser.parse_args()


#integer = flags.DEFINE_integer('viz_steps', default=400, help='Frequency at which save visualizations.')
#flags.DEFINE_integer('num_monte_carlo',
#                     default=50,
#                     help='Network draws to compute predictive probabilities.')
#flags.DEFINE_bool('fake_data',
#                  default=False,
#                  help='If true, uses fake data. Defaults to real data.')

#FLAGS = flags.FLAGS

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
            self.images, self.labels = [np.append(images[i].flatten(), labels[i])
                                        for i in range(np.shape(images)[0])],\
                                       labels
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


class DenseVariationalGrouped(tfkl.Layer):
    def __init__(self,
                 # input_size,
                 units,
                 kl_weight,
                 activation=None,
                 tau_0_inv=1000,
                 v=100,
                 num_groups=NUM_GROUPS,
                 num_sample=args.num_sample,
                 **kwargs):
        # self.input_size = input_size
        self.units = units
        # self.shape = (input_size, units)
        self.kl_weight = kl_weight
        self.activation = tfk.activations.get(activation)
        self.tau_0_inv = tau_0_inv
        self.v = v
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
        # np.sqrt(self.prior_pi_1 * self.prior_sigma_1 ** 2 +
        #        self.prior_pi_2 * self.prior_sigma_2 ** 2)

        super().__init__(**kwargs)

    #def compute_output_shape(self, input_shape):
    #    return input_shape[0], self.units

    def build(self, input_shape):
        print(input_shape)
        self.kernel_mu = self.add_weight(name='kernel_mu',
                                         shape=(input_shape[1] - 1, self.units),
                                         initializer=tf.random_normal_initializer(stddev=self.tau_0_inv),
                                         trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu',
                                       shape=(self.units,),
                                       initializer=tf.random_normal_initializer(stddev=self.tau_0_inv),
                                       trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho',
                                          shape=(input_shape[1] - 1, self.units),
                                          initializer=tf.random_normal_initializer(stddev=self.tau_0_inv),
                                          trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho',
                                        shape=(self.units,),
                                        initializer=tf.random_normal_initializer(stddev=self.tau_0_inv),
                                        trainable=True)
        self.kernel_mu_g = []
        self.bias_mu_g = []
        self.kernel_rho_g = []
        self.bias_rho_g = []
        self.tau_g = []
        self.gamma_g = []
        for i in range(self.num_groups):
            self.gamma_g.append(tfp.edward2.HalfNormal(scale=self.v))
            self.tau_g.append(tf.square(self.gamma_g[i]))
            self.kernel_mu_g.append(self.add_weight(name='kernel_mu_' + str(i),
                                                    shape=(input_shape[1] - 1, self.units),
                                                    initializer=tf.constant_initializer(self.kernel_mu.numpy()),
                                                    trainable=True))
            self.bias_mu_g.append(self.add_weight(name='bias_mu_' + str(i),
                                                  shape=(self.units,),
                                                  initializer=tf.constant_initializer(self.bias_mu.numpy()),
                                                  trainable=True))
            self.kernel_rho_g.append(self.add_weight(name='kernel_rho_' + str(i),
                                                     shape=(input_shape[1] - 1, self.units),
                                                     initializer=tf.random_normal_initializer(self.tau_g[i]),
                                                     trainable=True))
            self.bias_rho_g.append(self.add_weight(name='bias_rho_' + str(i),
                                                   shape=(self.units,),
                                                   initializer=tf.random_normal_initializer(stddev=self.tau_g[i]),
                                                   trainable=True))
        super().build(input_shape)

    def call(self, inputs, training=False, **kwargs):
        imgs = inputs[:, :-1]
        y_preds = tf.cast(inputs[:, -1], dtype=tf.int32)
        kernel_sigma = tf.math.softplus(self.kernel_rho)
        loss = 0.0

        #for i in range(self.num_sample):
        kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)
        kernel_g = []
        for i in range(self.num_groups):
            kernel_sigma_g = tf.math.softplus(self.kernel_rho_g[i])
            kernel_g.append(self.kernel_mu_g[i] + kernel_sigma_g * tf.random.normal(self.kernel_mu_g[i].shape))
        bias_sigma = tf.math.softplus(self.bias_rho)
        bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)
        bias_g = []
        for i in range(self.num_groups):
            bias_sigma_g = tf.math.softplus(self.bias_rho_g[i])
            bias_g.append(self.bias_mu_g[i] + bias_sigma_g * tf.random.normal(self.bias_mu_g[i].shape))

        loss += self.kl_weight * self.log_prior_prob(kernel, kernel_g, self.gamma_g)
        self.add_loss(loss / self.num_sample)

        kernel_g_pred = tf.gather(kernel_g, y_preds)
        bias_g_pred = tf.gather(bias_g, y_preds)

        if training:
            return self.activation(tfkb.dot(imgs, kernel) + bias)
            for i in range(1):#np.shape(imgs)[0]):
                if i == 0:
                    a = 0 #self.activation(tfkb.dot(imgs[i], kernel_g[y_preds[i]]) + bias_g[y_preds[i]])
                else:
                    a += 0 #self.activation(tfkb.dot(imgs[i], kernel_g[y_preds[i]]) + bias_g[y_preds[i]])

        else:
            return self.activation(tfkb.dot(imgs, kernel) + bias)
        # self.add_loss(self.kl_loss(kernel, self.kernel_mu, kernel_sigma) +
        #              self.kl_loss(bias, self.bias_mu, bias_sigma))

    '''
    def loss(target_y, predicted_y):
        self.wgtarget_y
        return tf.reduce_mean(tf.square(target_y - predicted_y))

    def train(model, inputs, outputs, learning_rate):
        with tf.GradientTape() as t:
            current_loss = loss(outputs, model(inputs))
        dW, db = t.gradient(current_loss, [model.W, model.b])
        model.W.assign_sub(learning_rate * dW)
        model.b.assign_sub(learning_rate * db)
    '''

    def kl_loss(self, w, w0, wg, mu, sigma):
        variational_dist = tfp.distributions.Normal(mu, sigma)
        return self.kl_weight * tfkb.sum(variational_dist.log_prob(w) - self.log_prior_prob(w0, wg))

    def log_prior_prob(self, w0, wg, gamma_g):
        w0_dist = tfp.distributions.Normal(tf.zeros_like(w0),
                                           tf.zeros_like(w0) + self.tau_0_inv)
        #b = int(self.num_groups)
        #tau_g_root = tfp.edward2.HalfNormal(scale=self.v) # int(self.num_groups))#.value
        #tau_g_root = tfp.distributions.Normal(tf.zeros(self.num_groups), self.v)  # .value
        #tau_g_root = tau_g_root * tau_g_root
        wg_dist = []
        loss_g = 0.0
        loss_tau_g = 0.0
        for i in range(self.num_groups):
            wg_dist.append(tfp.distributions.Normal(w0, tf.zeros_like(w0)))  # + tf.square(tau_g_root)))
            loss_g += tf.math.reduce_sum(wg_dist[i].log_prob(wg[i]))

            tau_g_distr = tfpd.Normal(loc=0, scale=self.v)
            loss_tau_g += tf.math.reduce_sum(tau_g_distr.log_prob(2 * gamma_g[i]))

        return tf.math.reduce_sum(w0_dist.log_prob(w0)) + loss_g + loss_tau_g


#def neg_log_likelihood(y_obs, y_pred, sigma=1):
#    dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
#    return tfkb.sum(-dist.log_prob(y_obs))


import warnings

warnings.filterwarnings('ignore')


#def f(x, sigma):
#    epsilon = np.random.randn(*x.shape) * sigma
#    return 10 * np.sin(2 * np.pi * (x)) + epsilon


prior_params = {
                "tau_0_inv": 100,
                "v": 1,
                "num_groups": 3,
                }

train_set, heldout_set = tf.keras.datasets.mnist.load_data(path='mnist.npz')

train_seq = MNISTSequence(data=train_set, batch_size=args.batch,
                          preprocessing=False)

input = tfk.Input(shape=(IMAGE_SHAPE[0] * IMAGE_SHAPE[1] + 1,), name='img')
output = DenseVariationalGrouped(NUM_CLASSES, 1/float(args.batch),
                                 **prior_params)(input)
model = tfk.Model(inputs=input, outputs=output)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tfk.optimizers.Adam(lr=args.lr),
              metrics=['mse']
             )

x = np.array(train_seq.images)
y = np.array(train_seq.labels, dtype=float)
y = train_seq.labels
print(model.summary())
#y = train_seq.labels
#model.evaluate(train.batch(BATCH_SIZE), steps=None, verbose=1)
model.fit(x, y, epochs=args.epochs, batch_size=train_seq.batch_size)
#model.predict(heldout_set.images)


if 0:
    train_size = 32
    noise = 1.0

    X = np.linspace(-0.5, 0.5, train_size).reshape(-1, 1)
    y = f(X, sigma=noise)
    y_true = f(X, sigma=0.0)

    plt.scatter(X, y, marker='+', label='Training data')
    plt.plot(X, y_true, label='Truth')
    plt.title('Noisy training data and ground truth')
    plt.legend()

    batch_size = train_size
    num_batches = train_size / batch_size

    kl_weight = 1.0 / num_batches
    prior_params = {
        'prior_sigma_1': 1.5,
        'prior_sigma_2': 0.1,
        'prior_pi': 0.5
    }

    x_in = tfkl.Input(shape=(1,))
    x = DenseVariationalGrouped(1, 20, kl_weight, **prior_params, activation='relu')(x_in)

    model = tfk.models.Model(x_in, x)


    def neg_log_likelihood(y_obs, y_pred, sigma=noise):
        dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
        return tfkb.sum(-dist.log_prob(y_obs))


    model.compile(loss=neg_log_likelihood(), optimizer=tfk.optimizers.Adam(lr=0.08), metrics=['mse'])
    model.fit(X, y, batch_size=batch_size, epochs=100, verbose=0)

    X_test = np.linspace(-1.5, 1.5, 1000).reshape(-1, 1)
    y_pred_list = []

    for i in tqdm.tqdm(range(500)):
        y_pred = model.predict(X_test)
        y_pred_list.append(y_pred)

    y_preds = np.concatenate(y_pred_list, axis=1)

    y_mean = np.mean(y_preds, axis=1)
    y_sigma = np.std(y_preds, axis=1)

    plt.plot(X_test, y_mean, 'r-', label='Predictive mean')
    plt.scatter(X, y, marker='+', label='Training data')
    plt.fill_between(X_test.ravel(),
                     y_mean + 2 * y_sigma,
                     y_mean - 2 * y_sigma,
                     alpha=0.5, label='Epistemic uncertainty')
    plt.title('Prediction')
    plt.legend()