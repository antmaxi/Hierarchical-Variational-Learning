# The code is partially adapted from
# http://krasserm.github.io/2019/03/14/bayesian-neural-networks/ and
# https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/bayesian_neural_network.py

# GPU_FLAG shows if the program will be run on GPU or local CPU

# the main arguments are
# num_epochs (for training the main network which decides class),
# num_sample (how many times to sample in Monte-Carlo sampling to estimated the integral
#             over the distribution for the main network)
# lr (learning rate for the main network)
# tau_inv_0 (initial and prior scale of weights)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys
import matplotlib.pyplot as plt
import math
import os
from utils import gpu_session
from absl import app

import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
import json
import argparse

import tensorflow as tf
import tensorflow_probability as tfp

from tensorboard.plugins.hparams import api as hp

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

GPU_FLAG = 0
if GPU_FLAG:
    LOG_DIR = '/local/home/antonma/HFL/my_tf_logs/my_tf_logs_gpu_non_'
else:
    LOG_DIR = os.getcwd() + '/results/my_tf_logs/my_tf_logs_non_'
LOG_DIR = LOG_DIR + datetime.now().strftime("%Y%m%d-%H%M%S")

parser = argparse.ArgumentParser(description='Choose the type of execution.')
parser.add_argument('--lr', help='Initial learning rate.', type=float,
                    default=0.001)
parser.add_argument('--num_epochs', help='Number of training steps to run.', type=int,
                    default=1)
parser.add_argument('--num_sample', help='Number of Monte-Carlo sampling repeats.', type=int,
                    default=5)
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

def get_shuffled_mnist(filename):
    train_set, heldout_set = tf.keras.datasets.mnist.load_data(path=filename)
    # get all the data concatenated
    imgs_all = np.concatenate((train_set[0], heldout_set[0]), axis=0)
    labels_all = np.concatenate((train_set[1], heldout_set[1]), axis=0)
    # choose indices for train set
    indices_train = np.arange(np.shape(imgs_all)[0])
    np.random.shuffle(indices_train)
    indices_test = indices_train[NUM_TRAIN_EXAMPLES:]
    indices_train = indices_train[0:NUM_TRAIN_EXAMPLES]
    train_set = (imgs_all[indices_train],
                 np.array(labels_all[indices_train], dtype=np.int)
                 )
    heldout_set = (imgs_all[indices_test],
                   np.array(labels_all[indices_test], dtype=np.int)
                   )

    return train_set, heldout_set

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

            # self.images = np.array([np.append(images[i].flatten(), labels[i]) for i in range(np.shape(images)[0])])
            self.images = np.array([images[i].flatten() for i in range(np.shape(images)[0])])
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


def train_model(model, train_seq, epochs=args.num_epochs,
                validation_data=None, validation_split=0.0,
                tensorboard_callback=None, hparams=None,
                typename="training"):
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
        log_dir=LOG_DIR + "/" + typename,
        histogram_freq=0,
        batch_size=args.batch,
        write_graph=True,
        write_grads=True,
        update_freq='epoch'
    )
    tensorboard.set_model(model)

    #earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    #mcp_save = tf.keras.callbacks.ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    #reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(
    #    monitor='val_loss', factor=0.1, patience=7, verbose=1,epsilon=1e-4, mode='min')

    if 0:
        def named_logs(model, logs):
            result = {}
            for l in zip(model.metrics_names, logs):
                result[l[0]] = l[1]
            return result

        for epoch in range(epochs):
            print('Epoch {}'.format(epoch))
            epoch_accuracy, epoch_loss = [], []
            for step, (batch_x, batch_y) in enumerate(train_seq):
                logs = model.train_on_batch(
                    batch_x, batch_y)
                tensorboard.on_epoch_end(step, named_logs(model, logs))

    else:
        callbacks = [tensorboard,]
        if not hparams is None:
            callbacks.append(hp.KerasCallback(LOG_DIR, hparams))

        if validation_data is None:
            training_history = model.fit(
                train_seq.images,  # input
                train_seq.labels,  # output
                batch_size=args.batch,
                verbose=1,
                epochs=epochs,
                validation_split=validation_split,
                callbacks=callbacks,#[tensorboard,
                           #earlyStopping,
                           #mcp_save,
                           #reduce_lr_loss],
            )
        else:
            training_history = model.fit(
                train_seq.images,  # input
                train_seq.labels,  # output
                batch_size=args.batch,
                verbose=1,
                epochs=epochs,
                validation_data=validation_data,
                callbacks=callbacks #[tensorboard,
                           #earlyStopping,
                           #mcp_save,
                           #reduce_lr_loss],
            )
    return model


class DenseVariationalNotGrouped(tfkl.Layer):
    def __init__(self,
                 units,
                 kl_weight,
                 activation=None,
                 tau_inv_0=None,
                 num_groups=NUM_GROUPS,
                 num_sample=args.num_sample,
                 **kwargs):
        self.units = units
        self.kl_weight = kl_weight
        self.activation = tfk.activations.get(activation)
        self.tau_inv_0 = tau_inv_0
        self.num_groups = num_groups
        self.num_sample = num_sample
        # Variational parameters
        self.kernel_mu = None
        self.bias_mu = None
        self.kernel_rho = None
        self.bias_rho = None
        # self.k = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        super().__init__(dynamic=False, **kwargs)

    #def compute_output_shape(self, input_shape):
    #    return input_shape[0], self.units

    def build(self, input_shape):
        print(input_shape)

        self.kernel_mu = self.add_weight(name='kernel_mu',
                                         shape=(input_shape[1], self.units),
                                         initializer=tf.constant_initializer(value=0),
                                         trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu',
                                       shape=(self.units,),
                                       initializer=tf.constant_initializer(value=0),
                                       trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho',
                                          shape=(input_shape[1], self.units),
                                          initializer=tf.constant_initializer(value=soft_inv(self.tau_inv_0 / 2.0)),
                                          trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho',
                                        shape=(self.units,),
                                        initializer=tf.constant_initializer(value=soft_inv(self.tau_inv_0 / 2.0)),
                                        trainable=True)
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

        imgs = inputs

        kernel_sigma = tf.math.softplus(self.kernel_rho) + eps
        bias_sigma = tf.math.softplus(self.bias_rho) + eps

        result = tf.zeros_like(tfkb.dot(imgs, self.kernel_mu))
        # Monte-Carlo for loss
        # TODO: is it possible not to calculate loss when predicting?

        # self.k.assign(0.0)
        elbo = tf.zeros(1, dtype=tf.float32)
        for step in range(self.num_sample):
            loss = 0.0
            kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)
            loss += self.kl_loss(kernel, self.kernel_mu, kernel_sigma, 0, self.tau_inv_0)

            bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)
            loss += self.kl_loss(bias, self.bias_mu, bias_sigma, 0, self.tau_inv_0)

            # throw out if loss is None (numerically too small argument of logarithm appeared somewhere)
            # if tf.math.is_nan(loss):
            #    loss = 0.0
            self.add_loss(loss / self.num_sample)
            elbo += loss
            # TODO: save ELBO

            result += self.activation(tfkb.dot(imgs, kernel) + bias) / self.num_sample
            tf.print(result)#, output_stream=sys.stderr)
        #tf.summary.scalar(elbo, "ELBO")
        #result = self.activation(tfkb.dot(imgs, self.kernel_mu) + bias)
        return result


import warnings


def main(argv):
    warnings.filterwarnings('ignore')
    #tf.summary.trace_on()
    #print(tf.executing_eagerly())

    prior_params = {
        "tau_inv_0": 1e-2,  # prior sigma of weights
    }
    #train_set, heldout_set = get_shuffled_mnist('mnist.npz')
    train_set, heldout_set = tf.keras.datasets.mnist.load_data(path='mnist.npz')

    train_seq = MNISTSequence(images=train_set[0], labels=train_set[1], batch_size=args.batch,
                              preprocessing=False)
    test_seq = MNISTSequence(images=heldout_set[0], labels=heldout_set[1], labels_to_binary=False,
                             batch_size=args.batch,
                             preprocessing=False)
    inputs = tfk.Input(shape=(IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
                              ), name='img')
    kl_weight = 1.0 / (NUM_TRAIN_EXAMPLES / float(args.batch))

    print(' ... Training main network')
    epochs = 100  # args.num_epochs$
    verbose = 0
    rel_counts = []
    uniques = []

    load_weights = 0  # to use trained model (load weights) or to train
    # group (from 0 to NUM_GROUPS-1)
    if load_weights:
        output = DenseVariationalNotGrouped(NUM_CLASSES, kl_weight,
                                            activation=None,
                                            **prior_params)(inputs)
        model = tfk.Model(inputs=inputs, outputs=output)
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      optimizer=tfk.optimizers.Adam(lr=args.lr),
                      metrics=['accuracy'],
                      # run_eagerly=True,
                      )
        # print(model.summary())

        trained_epochs = 100
        for g in range(NUM_GROUPS+1):
                foldername = "/home/anton/PycharmProjects/Hierarchical-Federated-Learning/results"  #"/home/anton/my_tf_logs/"
                #if trained_epochs == 1:
                #    name = "my_tf_logs_20201006-192650"
                #elif trained_epochs == 10:
                #    name = "my_tf_logs_20201006-194905"
                #else:
                #    return -1
                name = ""
                weights = np.load(foldername + name + "/weights_" + str(trained_epochs) + "_epochs.npy", allow_pickle=True)
                w = model.get_weights()
                if g == NUM_GROUPS:
                    off = 0
                else:
                    off = 6  # 6 for groups, 0 for main weights change
                w[0] = weights[off + 4 * g + 0]
                w[1] = weights[off + 4 * g + 1]
                w[2] = weights[off + 4 * g + 2]
                w[3] = weights[off + 4 * g + 3]
                model.set_weights(w)


                result_prob = model.predict(test_seq.images)
                result_argmax = np.argmax(result_prob, axis=1)

                print(sum(1 for x, y in zip(heldout_set[1], result_argmax) if x == y) / float(len(result_argmax)))
                res = np.zeros_like(result_argmax)
                for i in range(len(result_argmax)):
                    res[i] = int(heldout_set[1][i] + 1) if result_argmax[i] == heldout_set[1][i] \
                        else int(-1 - heldout_set[1][i])
                unique, counts = np.unique(res, return_counts=True)
                uniques.append(unique.astype(int))
                d = dict(zip(unique, counts))
                print("Distribution of correct/incorrect predictions (key - true label plus one")
                print(np.asarray((unique, counts)).T)
                all_indices, all_counts = np.unique(heldout_set[1], return_counts=True)
                a_all = dict(zip(all_indices, all_counts))
                rel_counts.append(np.array([d[i] / a_all[abs(i) - 1] for i in unique]))
                print(np.asarray((unique, rel_counts[g])).T)
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(8, 3))
        # fig.suptitle()
        # Accuracy of prediction depending on the true label for different types of weights
        ax0.bar(uniques[3][10:], rel_counts[3][10:])  # , tick_label=list(uniques[3][10:]))
        ax0.set_title("\"Zero\" weights")
        ax0.set_xticks(np.arange(1, 11))
        ax0.set_xticklabels(list(range(10)))
        ax1.bar(uniques[0][10:], rel_counts[0][10:])  # , tick_label=list(uniques[3][10:]))
        ax1.set_title("Group 1")
        ax1.set_xticks(np.arange(1, 11))
        ax1.set_xticklabels(list(range(10)))
        ax2.bar(uniques[1][10:], rel_counts[1][10:])  # , tick_label=list(uniques[3][10:]))
        ax2.set_title("Group 2")
        ax2.set_xticks(np.arange(1, 11))
        ax2.set_xticklabels(list(range(10)))
        ax3.bar(uniques[2][10:], rel_counts[2][10:])  # , tick_label=list(uniques[3][10:]))
        ax3.set_title("Group 3")
        ax3.set_xticks(np.arange(1, 11))
        ax3.set_xticklabels(list(range(10)))
        fig.tight_layout()
        plt.savefig("/home/anton/PycharmProjects/Hierarchical-Federated-Learning/results/" \
                    + str(trained_epochs) + "-epochs",
                    dpi=300)
    else:
        for m in range(5):
            for t in (1e-0,):# 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3):
                    for lr in (1e-1,):# 1e-1, 3e-2, 1e-2, 3e-3):
                        for num_sample in (20,):# 20, 50):
                            verbose = 0
                            prior_params = {
                                "tau_inv_0": t,  # prior sigma of weights
                                "num_sample": num_sample
                            }
                            print(epochs, num_sample, t, lr)
                            inputs1 = tf.keras.layers.Flatten()(inputs),

                            #output = tfp.layers.DenseFlipout(NUM_CLASSES,
                            #                                activation=None)(inputs)
                            output = DenseVariationalNotGrouped(NUM_CLASSES, kl_weight,
                                                                activation=None,
                                                                **prior_params)(inputs)
                            model = tfk.Model(inputs=inputs, outputs=output)
                            model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                                          optimizer=tfk.optimizers.Adam(lr=lr),
                                          metrics=['accuracy'],
                                          # run_eagerly=True,
                                          )

                            train_model(model, train_seq,
                                        validation_data=[test_seq.images,
                                                         tf.keras.utils.to_categorical(heldout_set[1],
                                                                                       num_classes=NUM_CLASSES)],
                                        epochs=epochs, hparams=None,
                                        typename="train_" + str(epochs) +  "_" + str(lr) + "_" + str(t) #+ "_" + str(num_sample)
                                        )


                            result_prob = model.predict(test_seq.images)
                            result_argmax = np.argmax(result_prob, axis=1)

                            acc = sum(1 for x, y in zip(heldout_set[1], result_argmax) if x == y) / float(len(result_argmax))
                            print(acc)
                            with open(LOG_DIR + "Output_non_hier.txt", "a+") as text_file:
                                text_file.write("{} {} {} {} \n".format(acc, t, lr, num_sample))

                        if verbose:
                            res = np.zeros_like(result_argmax)
                            for i in range(len(result_argmax)):
                                res[i] = int(heldout_set[1][i] + 1) if result_argmax[i] == heldout_set[1][i] \
                                    else int(-1 - heldout_set[1][i])
                            unique, counts = np.unique(res, return_counts=True)
                            uniques.append(unique.astype(int))
                            d = dict(zip(unique, counts))
                            print("Distribution of correct/incorrect predictions (key - true label plus one")
                            print(np.asarray((unique, counts)).T)
                            all_indices, all_counts = np.unique(heldout_set[1], return_counts=True)
                            a_all = dict(zip(all_indices, all_counts))
                            rel_counts.append(np.array([d[i] / a_all[abs(i) - 1] for i in unique]))
                            print(np.asarray((unique, rel_counts[0])).T)
    print(LOG_DIR)

if __name__ == '__main__':
    if GPU_FLAG:
        gpu_session(num_gpus=1)  #
    app.run(main)
