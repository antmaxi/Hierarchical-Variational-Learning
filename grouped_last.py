from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import os
from utils import gpu_session
from absl import app

from pathlib import Path
from datetime import datetime
import json
import argparse
import warnings

import tensorflow as tf
import tensorflow_probability as tfp

tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# tf.compat.v1.disable_eager_execution()

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
NUM_GROUPS = 2

gpu = 0
if gpu:
    LOG_DIR = '/local/home/antonma/HFL/my_tf_logs_'
else:
    LOG_DIR = '/home/anton/my_tf_logs/my_tf_logs_'
LOG_DIR = LOG_DIR + datetime.now().strftime("%Y%m%d-%H%M%S")

LABELS_CHANGE_DICT_GROUPED = {0: 0, 3: 0, 6: 0, 8: 0,  # 0
                              2: 1, 5: 1,  # 1
                              1: 1, 4: 1, 7: 1, 9: 1}  # 2

LABELS_CHANGE_GROUPED = []  # (0, 2, 1, 0, ...)
for i in range(NUM_CLASSES):
    LABELS_CHANGE_GROUPED.append(LABELS_CHANGE_DICT_GROUPED[i])
LABELS_CHANGE_GROUPED = tuple(LABELS_CHANGE_GROUPED)

parser = argparse.ArgumentParser(description='Choose the type of execution.')
parser.add_argument("--config_path",
                    type=Path,
                    help="Path to the main json config. "
                         "Ex: 'configurations/femnist_virtual.json'",
                    default='configurations/mnist_virtual.json')
parser.add_argument('--lr', help='Initial learning rate.', type=float,
                    default=0.001)
parser.add_argument('--num_epochs', help='Number of training steps to run.', type=int,
                    default=10)
parser.add_argument('--num_epochs_g', help='Number of training steps to run grouped inference.', type=int,
                    default=5)
parser.add_argument('--num_sample', help='Number of Monte-Carlo sampling repeats.', type=int,
                    default=10)
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

    def __init__(self, images=None, labels=None,
                 labels_to_binary=True, batch_size=args.batch,
                 used_labels=tuple(range(10)), labels_len=NUM_CLASSES,
                 labels_change=tuple(range(10)), preprocessing=False, fake_data_size=None):
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
            self.images, self.labels_int, self.labels_bin = MNISTSequence.__preprocessing(
                images, labels, used_labels, labels_change, labels_len)
        else:
            images = 2 * (images / 255.) - 1.
            self.labels_bin = tf.keras.utils.to_categorical(labels)
            self.labels_int = labels
            self.images = np.array(
                [images[i].flatten() for i in range(np.shape(images)[0])])

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
        labels_int = np.random.randint(low=0, high=num_classes,
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
                labels_transformed.append(int(labels_change[labels[index]]))
        labels_transformed = np.array(labels_transformed, dtype=np.int32)
        # Normalize images
        images = 2 * (images[indices] / 255.) - 1.
        images = images[..., tf.newaxis]
        # Convert labels for cross-entropy loss
        labels_bin = tf.keras.utils.to_categorical(y=labels_transformed, num_classes=labels_len)
        labels_int = labels_transformed
        return images, labels_int, labels_bin

    def __len__(self):
        return int(math.ceil(np.shape(self.images)[0] / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


def create_group_inference_model(type="dens1", output_size=NUM_CLASSES):
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


def train_model(model, train_seq, epochs=args.num_epochs, verbose=0, heldout_seq=((), ()), use_tensorboard=True):
    """
    Trains LeNet model on MNIST data in a flexible to data way

    :param epochs:
    :param model:
    :param train_seq:
    :param heldout_seq:
    :param tensorboard_callback:
    :return: trained model
    """
    if use_tensorboard:
        tensorboard = tfk.callbacks.TensorBoard(
            log_dir=LOG_DIR,
            histogram_freq=0,
            write_graph=True,
            write_grads=True
        )
        tensorboard.set_model(model)
    else:
        tensorboard = None

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
            [train_seq.images, train_seq.labels_bin, train_seq.labels_int],  # input
            train_seq.labels_bin,  # output
            validation_split=0.2,
            batch_size=args.batch,
            verbose=verbose,
            epochs=epochs,
            # validation_data=(x_test, y_test),
            callbacks=[tensorboard],
        )
    return model


class DenseVariationalGrouped(tfkl.Layer):
    def __init__(self,
                 units,
                 kl_weight,
                 training=False,
                 clip=False,
                 activation=None,
                 tau_inv_0=None,
                 v=None,
                 num_groups=NUM_GROUPS,
                 num_sample=args.num_sample,
                 **kwargs):
        self.training = training
        self.units = units
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
        self.clip = clip
        self.k = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.kernel_mu = self.add_weight(name='kernel_mu',
                                         shape=(input_shape[0][1], self.units),
                                         initializer=tf.constant_initializer(value=0.0),
                                         trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu',
                                       shape=(self.units,),
                                       initializer=tf.constant_initializer(value=0.0),
                                       trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho',
                                          shape=(input_shape[0][1], self.units),
                                          initializer=tf.constant_initializer(value=soft_inv(self.tau_inv_0)),
                                          trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho',
                                        shape=(self.units,),
                                        initializer=tf.constant_initializer(value=soft_inv(self.tau_inv_0)),
                                        trainable=True)
        self.gamma_mu = self.add_weight(name='gamma_mu',
                                        shape=(self.num_groups,),
                                        initializer=tf.constant_initializer(value=0),
                                        trainable=True)
        self.gamma_rho = self.add_weight(name='gamma_rho',
                                         shape=(self.num_groups,),
                                         initializer=tf.constant_initializer(value=soft_inv(self.v)),
                                         trainable=True)
        self.kernel_mu_g = []
        self.bias_mu_g = []
        self.kernel_rho_g = []
        self.bias_rho_g = []
        for i in range(self.num_groups):
            self.kernel_mu_g.append(self.add_weight(name='kernel_mu_' + str(i),
                                                    shape=(input_shape[0][1], self.units),
                                                    initializer=tf.constant_initializer(value=0.0),
                                                    trainable=True))
            self.bias_mu_g.append(self.add_weight(name='bias_mu_' + str(i),
                                                  shape=(self.units,),
                                                  initializer=tf.constant_initializer(value=0.0),
                                                  trainable=True))
            self.kernel_rho_g.append(self.add_weight(name='kernel_rho_' + str(i),
                                                     shape=(input_shape[0][1], self.units),
                                                     initializer=tf.constant_initializer(
                                                         value=soft_inv(self.tau_inv_0)),
                                                     trainable=True))
            self.bias_rho_g.append(self.add_weight(name='bias_rho_' + str(i),
                                                   shape=(self.units,),
                                                   initializer=tf.constant_initializer(value=soft_inv(self.tau_inv_0)),
                                                   trainable=True))
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

        imgs = inputs[0]  #:, : -NUM_CLASSES - 1]
        y_preds = tf.squeeze(tf.cast(inputs[2], dtype=tf.int32), axis=1)  #:, -NUM_CLASSES - 1:-1]

        kernel_sigma = tf.math.softplus(self.kernel_rho) + eps
        bias_sigma = tf.math.softplus(self.bias_rho) + eps

        kernel_sigma_g = []
        for i in range(self.num_groups):
            kernel_sigma_g.append(tf.math.softplus(self.kernel_rho_g[i]) + eps)
        bias_sigma_g = []
        for i in range(self.num_groups):
            bias_sigma_g.append(tf.math.softplus(self.bias_rho_g[i]) + eps)
        gamma_sigma = tf.math.softplus(self.gamma_rho) + eps

        n = imgs.get_shape().as_list()[0]
        result = 1e-8 * tf.ones_like(tfkb.dot(imgs, self.kernel_mu))

        # Monte-Carlo for loss
        # TODO: is it possible not to calculate loss when predicting?

        self.k.assign(0.0)
        count = tf.constant(0.0)
        imgs = tf.expand_dims(imgs, axis=1)  # batch x 1 x 784
        for step in range(self.num_sample):
            loss = 0.0
            kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)
            if training:
                loss += self.kl_loss(kernel, self.kernel_mu, kernel_sigma, 0, self.tau_inv_0)

            bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)
            loss += self.kl_loss(bias, self.bias_mu, bias_sigma, 0, self.tau_inv_0)

            # print(tfkb.get_value(bias_sigma_g))
            # tau_inv_g = tf.ones_like(tau_inv_g)
            # print(tfkb.get_value(tau_inv_g))

            gamma_g = self.gamma_mu + gamma_sigma * tf.random.normal(self.gamma_mu.shape)
            # tf.print(gamma_g)
            min_gamma = 1
            loss += self.kl_loss(gamma_g, self.gamma_mu, gamma_sigma, 0, self.v)

            tau_inv_g = tf.square(gamma_g)
            # TODO: how to prevent NaNs in loss without clipping?
            if self.clip:
                tau_inv_g = tf.clip_by_value(tau_inv_g, self.v * self.v / 1e2, 1e7)

            kernel_g = []
            for i in range(self.num_groups):
                kernel_g.append(self.kernel_mu_g[i] + kernel_sigma_g[i] * tf.random.normal(self.kernel_mu_g[i].shape))
                loss += self.kl_loss(kernel_g[i], self.kernel_mu_g[i], kernel_sigma_g[i], kernel, tau_inv_g[i])

            bias_g = []
            for i in range(self.num_groups):
                bias_g.append(self.bias_mu_g[i] + bias_sigma_g[i] * tf.random.normal(self.bias_mu_g[i].shape))
                loss += self.kl_loss(bias_g[i], self.bias_mu_g[i], bias_sigma_g[i], bias, tau_inv_g[i])
            # throw out if loss is None (numerically too small argument of logarithm appeared somewhere)
            # loss = tf.constant(1.0) / tf.constant(0.0)
            loss = tfkb.switch(tf.math.is_nan(loss), 0.0, loss)
            loss = tfkb.switch(tf.math.is_inf(loss), 0.0, loss)
            loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
            # def f(gamma_g, level):
            #    return tf.reduce_all(tf.math.greater(tf.math.abs(gamma_g), level * tf.ones_like(gamma_g)))

            # loss = tfkb.switch(f(gamma_g, 0.1),
            #                   loss, 0.0)
            # doesn't work
            # self.k.assign(tfkb.switch(loss, tf.math.add(self.k, 1.0), tf.math.add(self.k, 0.0)))
            #
            # tfkb.switch(tf.math.is_nan(loss), tf.math.add(count, 0.0), tf.math.add(count, 1.0))

            self.add_loss(loss / self.num_sample)

            train_summary_writer = tf.summary.create_file_writer(LOG_DIR)
            tf.summary.histogram(
                'gamma_rho',
                self.gamma_rho,
            )
            tf.summary.histogram(
                'gamma_mu',
                self.gamma_mu,
            )

            if training:
                kernel_g_pred = tf.gather(kernel_g, y_preds, axis=0)  # batch x 784 x units
                bias_g_pred = tf.gather(bias_g, y_preds, axis=0)  # batch x units
                # if training:  # and (not n is None):
                # imgs - batch x 1 x 784
                # TODO: more efficient this part?
                d = tf.linalg.matmul(imgs, kernel_g_pred)
                d = tf.squeeze(d, axis=1)
                result += self.activation(d + bias_g_pred)
            else:
                aux = tf.zeros_like(result)
                for j in range(NUM_GROUPS):
                    aux += tf.math.multiply(tfkb.dot(imgs, kernel_g[j]) + bias_g[j],
                                            tf.tile(tf.expand_dims(inputs[0][:, j], 1),
                                                    tf.constant([1, self.units], tf.int32))
                                            )
                result += self.activation(aux)
        return result / self.num_sample  # TODO: normalize to number of non-NaNs?
        #    result += self.activation(tfkb.dot(imgs, kernel) + bias) / self.num_sample
        #
        #   aux = tfkb.switch(tf.math.is_nan(loss), tf.zeros_like(result), aux)
        #   aux = tfkb.switch(loss, aux, tf.zeros_like(result))
        # #tfkb.switch(count > 0, y, tf.zeros_like(result))


def create_compile_class_inference_model(num_sample=5, lr=0.01, clip=False, prior_params=None,
                                         training=None):
    input_img = tfk.Input(shape=(IMAGE_SHAPE[0] * IMAGE_SHAPE[1],), name='img')
    input_logits = tfk.Input(shape=(NUM_CLASSES,), name='logits')
    input_int = tfk.Input(shape=(1,), name='int_class')
    combined_input = [input_img, input_logits, input_int]

    kl_weight = 1.0 / (NUM_TRAIN_EXAMPLES / float(args.batch)) / 100
    # kl_weight = 0.0
    output1 = DenseVariationalGrouped(100, kl_weight,
                                      activation=None,
                                      num_sample=num_sample,
                                      **prior_params,
                                      clip=clip)(combined_input, training=training)
    output2 = DenseVariationalGrouped(NUM_CLASSES, kl_weight,
                                      activation="softmax",
                                      num_sample=num_sample,
                                      **prior_params,
                                      clip=clip)([output1, input_logits, input_int], training=True)
    model = tfk.Model(inputs=combined_input, outputs=output2)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  optimizer=tfk.optimizers.Adam(lr=lr),
                  metrics=['accuracy'],
                  run_eagerly=True,
                  )
    return model


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
        "tau_inv_0": 1e-2,  # prior scale of weights dispersion around 0
        "v": 1e-4,  # prior scale of group weights dispersion around main ones
        "num_groups": NUM_GROUPS,
    }

    train_set, heldout_set = tf.keras.datasets.mnist.load_data(path='mnist.npz')

    train_seq = MNISTSequence(images=train_set[0], labels=train_set[1], batch_size=args.batch,
                              preprocessing=False, labels_to_binary=False)

    # GROUP inference model
    model_grouped = create_group_inference_model(type="lenet", output_size=NUM_GROUPS)
    train_seq_grouped = MNISTSequence(images=train_set[0], labels=train_set[1], batch_size=args.batch,
                                      labels_change=LABELS_CHANGE_GROUPED, labels_len=NUM_GROUPS,
                                      preprocessing=True)
    print(' ... Training group inference network')
    # train_model(model_grouped, train_seq_grouped, epochs=args.num_epochs_g)
    epochs = 2
    num_sample = 5
    pretrain = 0
    for t in (1e-0, 1e-1,):
        for v in (1e-0, 1e-1,):
            for lr in (1e-1, 1e-2, 1e-3):
                for clip in (False,):
                    prior_params["tau_inv_0"] = t
                    prior_params["v"] = v

                    model = create_compile_class_inference_model(num_sample=num_sample, clip=clip, lr=lr,
                                                                 prior_params=prior_params,
                                                                 training=True
                                                                 )
                    train_seq.labels_int = train_seq_grouped.labels_int
                    if pretrain:
                        model_dense = tfk.Sequential(
                            tfkl.Dense(NUM_CLASSES)
                        )
                        model_dense.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                                            optimizer=tfk.optimizers.Adam(lr=lr),
                                            metrics=['accuracy'],
                                            run_eagerly=True,
                                            )
                        model_dense.fit(train_seq.images, train_seq.labels_bin, epochs=1, verbose=1)
                        weights_pretrained = model_dense.get_weights()
                        # print(model.summary())
                        weights_initial = model.get_weights()
                        weights_initial[0] = weights_pretrained[0]
                        weights_initial[6] = weights_pretrained[0]
                        weights_initial[8] = weights_pretrained[0]
                        weights_initial[10] = weights_pretrained[0]
                        weights_initial[1] = weights_pretrained[1]
                        weights_initial[7] = weights_pretrained[1]
                        weights_initial[9] = weights_pretrained[1]
                        weights_initial[11] = weights_pretrained[1]
                        model.set_weights(weights_initial)

                    print(' ... Training main network')
                    train_model(model, train_seq, epochs=epochs, verbose=1, )

                    model_predict = create_compile_class_inference_model(num_sample=num_sample, clip=clip, lr=lr,
                                                                         prior_params=prior_params,
                                                                         training=False
                                                                         )
                    model_predict.set_weights(model.get_weights())
                    predicted_classes = model_predict.predict(
                        x=[train_seq.images, train_seq.labels_bin, train_seq.labels_int],
                        batch_size=None, verbose=1)
                    print(clip, t, v, lr)
                    result_argmax = np.argmax(predicted_classes, axis=1)
                    print(sum(1 for x, y in zip(train_set[1], result_argmax) if x == y) / float(len(result_argmax)))
                    res = np.zeros_like(result_argmax)
                    for i in range(len(result_argmax)):
                        res[i] = result_argmax[i] + 1 if result_argmax[i] == train_set[1][i] else -result_argmax[i] - 1
                    unique, counts = np.unique(res, return_counts=True)
                    print(np.asarray((unique, counts)).T)
                    ## Predict groups
                    # heldout_seq_grouped = MNISTSequence(images=heldout_set[0], labels=heldout_set[1], batch_size=args.batch,
                    #                                    labels_change=LABELS_CHANGE_GROUPED, labels_len=NUM_GROUPS,
                    #                                    preprocessing=True)
                    # train_model(model_grouped, train_seq_grouped, epochs=args.num_epochs_g)
                    # print(" ... Predicting groups")
                    # predicted_groups_probs = model_grouped.predict(x=heldout_seq_grouped.images, batch_size=None, verbose=1)
                    # predicted_groups_probs = np.hstack((predicted_groups_probs,
                    #                                    np.zeros((np.shape(predicted_groups_probs)[0],
                    #                                              NUM_CLASSES - NUM_GROUPS))))
                    ## images with appended
                    # test_seq = MNISTSequence(images=heldout_set[0], labels=predicted_groups_probs, labels_to_binary=False,
                    #                         batch_size=args.batch,
                    #                         preprocessing=False)
                    # result_prob = model.predict(test_seq.images)
                    # result_argmax = np.argmax(result_prob, axis=1)
                    # print(clip, t, v)
                    # print(sum(1 for x, y in zip(heldout_set[1], result_argmax) if x == y) / float(len(result_argmax)))
                    # res = np.zeros_like(result_argmax)
                    # for i in range(len(result_argmax)):
                    #    res[i] = result_argmax[i] + 1 if result_argmax[i] == heldout_set[1][i] else -result_argmax[i] - 1
                    # unique, counts = np.unique(res, return_counts=True)
                    # print(np.asarray((unique, counts)).T)


if __name__ == '__main__':
    if gpu:
        gpu_session(num_gpus=1)  #
    app.run(main)
