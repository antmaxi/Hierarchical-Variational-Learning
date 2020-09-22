from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import math
import os
from utils import gpu_session
from absl import app

from datetime import datetime
# from time import time
# from tensorboard.python.keras.callbacks import TensorBoard
import tensorflow as tf
import tensorflow_probability as tfp

import argparse

tf.compat.v1.disable_eager_execution()

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
parser.add_argument('--lr', help='Initial learning rate.', type=float,
                    default=0.001)
parser.add_argument('--num_epochs', help='Number of training steps to run.', type=int,
                    default=5)
parser.add_argument('--num_epochs_g', help='Number of training steps to run grouped inference.', type=int,
                    default=5)
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


# integer = flags.DEFINE_integer('viz_steps', default=400, help='Frequency at which save visualizations.')
# flags.DEFINE_integer('num_monte_carlo',
#                     default=50,
#                     help='Network draws to compute predictive probabilities.')
# flags.DEFINE_bool('fake_data',
#                  default=False,
#                  help='If true, uses fake data. Defaults to real data.')

# FLAGS = flags.FLAGS

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


def create_model(type="dens_one", output_size=NUM_CLASSES):
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

    :param model:
    :param train_seq:
    :param heldout_seq:
    :param tensorboard_callback:
    :return: trained model
    """

    print(' ... Training neural network')
    for epoch in range(epochs):
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
                    batch_loss,
                    batch_accuracy))
                # tf.reduce_mean(),
                # tf.reduce_mean()))
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


class DenseVariationalGrouped(tfkl.Layer):
    def __init__(self,
                 # input_size,
                 units,
                 kl_weight,
                 activation=None,
                 tau_inv_0=1.0,
                 v=1.0,
                 num_groups=NUM_GROUPS,
                 num_sample=args.num_sample,
                 **kwargs):
        # self.input_size = input_size
        self.units = units
        # self.shape = (input_size, units)
        self.kl_weight = kl_weight
        self.activation = tfk.activations.get(activation)
        self.tau_inv_0 = tau_inv_0
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
        self.gamma_mu = None
        self.gamma_rho = None
        # np.sqrt(self.prior_pi_1 * self.prior_sigma_1 ** 2 +
        #        self.prior_pi_2 * self.prior_sigma_2 ** 2)

        super().__init__(**kwargs)

    # def compute_output_shape(self, input_shape):
    #    return input_shape[0], self.units

    def build(self, input_shape):
        print(input_shape)

        softplus = tfp.bijectors.Softplus()
        self.kernel_mu = self.add_weight(name='kernel_mu',
                                         shape=(input_shape[1] - NUM_CLASSES, self.units),
                                         initializer=tf.random_normal_initializer(stddev=self.tau_inv_0),
                                         trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu',
                                       shape=(self.units,),
                                       initializer=tf.random_normal_initializer(stddev=self.tau_inv_0),
                                       trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho',
                                          shape=(input_shape[1] - NUM_CLASSES, self.units),
                                          initializer=tf.random_normal_initializer(stddev=self.tau_inv_0),
                                          trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho',
                                        shape=(self.units,),
                                        initializer=tf.random_normal_initializer(
                                                    stddev=self.tau_inv_0), #softplus.inverse(tf.convert_to_tensor(self.tau_inv_0, dtype=tf.float32))),
                                        trainable=True)
        self.gamma_mu = self.add_weight(name='gamma_mu',
                                        shape=(self.num_groups,),
                                        initializer=tf.random_normal_initializer(stddev=self.v),
                                        trainable=True)
        self.gamma_rho = self.add_weight(name='gamma_rho',
                                         shape=(self.num_groups,),
                                         initializer=tf.random_normal_initializer(stddev=self.v),
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
                                                    shape=(input_shape[1] - NUM_CLASSES, self.units),
                                                    initializer=tf.random_normal_initializer(stddev=self.tau_inv_0),
                                                    # tf.constant_initializer(self.kernel_mu.numpy()),
                                                    # tf.random_normal_initializer(stddev=self.tau_inv_0),
                                                    trainable=True))
            self.bias_mu_g.append(self.add_weight(name='bias_mu_' + str(i),
                                                  shape=(self.units,),
                                                  initializer=tf.random_normal_initializer(stddev=self.tau_inv_0),
                                                  # tf.constant_initializer(self.bias_mu.numpy()),
                                                  # tf.random_normal_initializer(stddev=self.tau_inv_0), #
                                                  trainable=True))
            self.kernel_rho_g.append(self.add_weight(name='kernel_rho_' + str(i),
                                                     shape=(input_shape[1] - NUM_CLASSES, self.units),
                                                     initializer=tf.random_normal_initializer(stddev=self.tau_g[i]),
                                                     trainable=True))
            self.bias_rho_g.append(self.add_weight(name='bias_rho_' + str(i),
                                                   shape=(self.units,),
                                                   initializer=tf.random_normal_initializer(stddev=self.tau_g[i]),
                                                   trainable=True))
        super().build(input_shape)

    def kl_loss(self, w, mu_var, sigma_var, mu_prior, sigma_prior):
        variational_dist = tfp.distributions.Normal(mu_var, sigma_var)
        a = self.kl_weight * tfkb.sum(variational_dist.log_prob(w) - self.log_prior_prob(w, mu_prior, sigma_prior))
        return a

    def log_prior_prob(self, w, mu, sigma):
        prior_dist = tfp.distributions.Normal(tf.zeros_like(w) + mu, tf.zeros_like(w) + sigma)
        a = prior_dist.log_prob(w) # tf.math.reduce_sum(
        return a

    def call(self, inputs, training=False, **kwargs):

        eps = 0#1e-5

        imgs = inputs[:, : -NUM_CLASSES]

        y = inputs[:, -NUM_CLASSES - 1:-1]

        y_preds = TABLE.lookup(tf.cast(tf.math.argmax(y, axis=1), dtype=tf.int32))  # get groups from classes

        # y_grouped = tf.zeros_like(inputs[:, -NUM_CLASSES - 1:-1])  # batch x NUM_GROUPS  (q(z|x))
        # y_grouped = tf.gather(kernel_g, y_preds, axis=0)  # batch x 784 x 10
        # for j in range(NUM_CLASSES):
        #    y_grouped[:, LABELS_CHANGE_DICT_GROUPED[j]] += y[:, j]

        kernel_sigma = tf.math.softplus(self.kernel_rho) + eps
        bias_sigma = tf.math.softplus(self.bias_rho) + eps
        print("???")
        print(tfkb.get_value(self.bias_rho))
        print(tfkb.get_value(self.kernel_rho))
        print(tfkb.get_value(self.gamma_rho))
        print(tfkb.get_value(bias_sigma))
        kernel_sigma_g = []
        for i in range(self.num_groups):
            kernel_sigma_g.append(tf.math.softplus(self.kernel_rho_g[i]) + eps)
        bias_sigma_g = []
        for i in range(self.num_groups):
            bias_sigma_g.append(tf.math.softplus(self.bias_rho_g[i]) + eps)
        gamma_sigma = tf.math.softplus(self.gamma_rho) + eps

        n = imgs.get_shape().as_list()[0]
        result = tf.zeros_like(tfkb.dot(imgs, self.kernel_mu))
        # Monte-Carlo for loss
        # TODO: is it possible not to calculate loss when predicting?
        for step in range(self.num_sample):
            loss = 0.0
            kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)
            loss += self.kl_loss(kernel, self.kernel_mu, kernel_sigma, 0, self.tau_inv_0)

            bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)
            loss += self.kl_loss(bias, self.bias_mu, bias_sigma, 0, self.tau_inv_0)

            gamma_g = self.gamma_mu + gamma_sigma * tf.random.normal(self.gamma_mu.shape)
            loss += self.kl_loss(gamma_g, self.gamma_mu, gamma_sigma, 0, self.v)

            tau_inv_g = tf.square(gamma_g)

            kernel_g = []
            for i in range(self.num_groups):
                kernel_g.append(self.kernel_mu_g[i] + kernel_sigma_g[i] * tf.random.normal(self.kernel_mu_g[i].shape))
                loss += self.kl_loss(kernel_g[i], self.kernel_mu_g[i], kernel_sigma_g[i], kernel, tau_inv_g[i])

            bias_g = []
            for i in range(self.num_groups):
                bias_g.append(self.bias_mu_g[i] + bias_sigma_g[i] * tf.random.normal(self.bias_mu_g[i].shape))
                loss += self.kl_loss(bias_g[i], self.bias_mu_g[i], bias_sigma_g[i], bias, tau_inv_g[i])
            print(loss)

            self.add_loss(loss / self.num_sample)
            # print(loss / self.num_sample)
            ####################   OUTPUT   ###################

            kernel_g_pred = tf.gather(kernel_g, y_preds, axis=0)  # batch x 784 x 10
            bias_g_pred = tf.gather(bias_g, y_preds, axis=0)  # batch x 10

            if training:  # and (not n is None):
                # imgs - batch x 784
                d = tf.tensordot(imgs, kernel_g_pred, axes=[[1], [1]])
                d = tf.transpose(d, perm=[2, 1, 0])
                d = tf.linalg.diag_part(d)
                d = tf.transpose(d, perm=[1, 0])
                aux = self.activation(d + bias_g_pred) / self.num_sample
                # print(aux)
                result += aux
            else:
                # result = self.activation(tfkb.dot(imgs, kernel) + bias)
                for j in range(NUM_GROUPS):
                    result += tf.math.multiply(tfkb.dot(imgs, kernel_g[j]) + bias_g[j],
                                               tf.tile(tf.expand_dims(y[:, j], 1),
                                                       tf.constant([1, NUM_CLASSES], tf.int32))
                                               )
                #    aux =
                #    result += aux

        return result


# def neg_log_likelihood(y_obs, y_pred, sigma=1):
#    dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
#    return tfkb.sum(-dist.log_prob(y_obs))


import warnings


def main(argv):
    warnings.filterwarnings('ignore')

    #tf.debugging.experimental.enable_dump_debug_info("/tmp/tfdbg2_logdir", tensor_debug_mode="FULL_HEALTH",
    #    circular_buffer_size=-1)

    prior_params = {
        "tau_inv_0": 1.0,
        "v": 1.0,
        "num_groups": 3,
    }

    train_set, heldout_set = tf.keras.datasets.mnist.load_data(path='mnist.npz')

    train_seq = MNISTSequence(images=train_set[0], labels=train_set[1], batch_size=args.batch,
                              preprocessing=False)

    inputs = tfk.Input(shape=(IMAGE_SHAPE[0] * IMAGE_SHAPE[1] + NUM_CLASSES,), name='img')
    output = DenseVariationalGrouped(NUM_CLASSES, 1.0 / (NUM_TRAIN_EXAMPLES / float(args.batch)), activation="relu",
                                     **prior_params)(inputs)
    model = tfk.Model(inputs=inputs, outputs=output)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  optimizer=tfk.optimizers.Adam(lr=args.lr),
                  metrics=['mse']
                  )
    model.build(input_shape=[None, IMAGE_SHAPE[0] * IMAGE_SHAPE[1] + NUM_CLASSES])
    # x = np.array(train_seq.images)
    # y = np.reshape(train_seq.labels, (-1, 1))
    print(model.summary())
    # model.evaluate(train.batch(BATCH_SIZE), steps=None, verbose=1)
    # model.fit(x, y, epochs=args.num_epochs, batch_size=train_seq.batch_size)
    print(' ... Training main network')
    train_model(model, train_seq, epochs=args.num_epochs)

    # GROUP inference model
    model_grouped = create_model(type="dens1", output_size=NUM_GROUPS)
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
    result_prob = model.predict(test_seq[0])
    result_argmax = np.argmax(result_prob, axis=1)
    print(sum(1 for x, y in zip(heldout_set[1], result_argmax) if x == y) / float(len(result_argmax)))

    if 0:
        def neg_log_likelihood(y_obs, y_pred, sigma=noise):
            dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
            return tfkb.sum(-dist.log_prob(y_obs))



if __name__ == '__main__':
    #gpu_session(num_gpus=1)  #
    app.run(main)
