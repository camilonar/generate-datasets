"""
Convert the trfrecords files in images with a specific format: class_numberofimage.jpg 
"""
import sys
import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError

from libs.incremental_eureka.training.config.megabatch_config import MegabatchConfig
from libs.incremental_eureka.training.config.general_config import GeneralConfig
from libs.incremental_eureka.etl.data.mnist_data import MnistData
from libs.incremental_eureka.etl.data.cifar_data import CifarData
from libs.incremental_eureka.etl.data.fashion_mnist_data import FashionMnistData
from libs.incremental_eureka.utils.train_modes import TrainMode
from libs.incremental_eureka.utils.default_paths import *
import numpy as np

import scipy.misc


def _int64_feature(value: int) -> tf.train.Features.FeatureEntry:
    """Create a Int64List Feature

    Args:
        value: The value to store in the feature

    Returns:
        The FeatureEntry
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value: str) -> tf.train.Features.FeatureEntry:
    """Create a BytesList Feature

    Args:
        value: The value to store in the feature

    Returns:
        The FeatureEntry
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_config(n_megabatchs):
    general_config = GeneralConfig(TrainMode.INCREMENTAL, 0.0001)
    for _ in range(n_megabatchs):
        train_conf = MegabatchConfig(1, batch_size=128)
        general_config.add_train_conf(train_conf)
    return general_config


def convert_to_list_mnist(folder):
    n_megabatchs = 5
    general_config = create_config(n_megabatchs)
    paths = get_paths_from_dataset(const.DATA_MNIST)
    pipeline = MnistData(general_config, paths[0], paths[1])

    def _process_examples(images, labels, itera, folder):
        for index in range(len(images)):
            num_image = itera + index  # TODO: mirar si no hay problema que el numero de la imagen no sea seguido
            output = '{}/{}_{}.png'.format(folder, labels[index], num_image)
            scipy.misc.imsave(output, np.squeeze(images[index]))

    conver_to_list(pipeline, _process_examples, folder)


def convert_to_list_cifar(folder):
    n_megabatchs = 5
    general_config = create_config(n_megabatchs)
    paths = get_paths_from_dataset(const.DATA_CIFAR_10)
    pipeline = CifarData(general_config, paths[0], paths[1])

    def _process_examples(images, labels, itera, folder):
        for index in range(len(images)):
            num_image = itera + index  # TODO: mirar si no hay problema que el numero de la imagen no sea seguido
            output = '{}/{}_{}.png'.format(folder, labels[index], num_image)

            print(images[index].shape)
            scipy.misc.imsave(output, images[index])
    conver_to_list(pipeline, _process_examples, folder)

def convert_to_list_fashion(folder):
    n_megabatchs = 5
    general_config = create_config(n_megabatchs)
    paths = get_paths_from_dataset(const.DATA_FASHION_MNIST)
    pipeline = FashionMnistData(general_config, paths[0], paths[1])

    def _process_examples(images, labels, itera, folder):
        for index in range(len(images)):
            num_image = itera + index  # TODO: mirar si no hay problema que el numero de la imagen no sea seguido
            output = '{}/{}_{}.png'.format(folder, labels[index], num_image)
            print(images[index].shape)
            scipy.misc.imsave(output, np.squeeze(images[index]))

    conver_to_list(pipeline, _process_examples, folder)

def conver_to_list(pipeline, _process_examples, output_folder):
    sess = tf.InteractiveSession()
    for i in range(len(pipeline.general_config.train_configurations)):
        pipeline.change_dataset_part(i)
        training_iterator, data_x, data_y = pipeline.build_train_data_tensor()
        sess.run(training_iterator.initializer)
        itera = 0
        while True:
            try:
                image_batch, target_batch = sess.run([data_x, data_y])
                _process_examples(image_batch, target_batch, itera,output_folder)
                # Process samples
                itera += 1
            except OutOfRangeError:
                break


if __name__ == '__main__':
    convert_to_list_fashion("dataset")
