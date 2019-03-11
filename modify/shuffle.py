"""
Shuffles a TFRecord file
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


def shuffle_mnist(output_name):
    n_megabatchs = 5
    general_config = create_config(n_megabatchs)
    paths = get_paths_from_dataset(const.DATA_MNIST)
    pipeline = MnistData(general_config, paths[0], paths[1])

    def _process_examples(images, labels, writer):
        _, rows, cols, depth = images.shape
        for index in range(len(images)):
            sys.stdout.write(f"\rProcessing sample {index + 1}")
            sys.stdout.flush()
            image_raw = images[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'label': _int64_feature(int(labels[index])),
                'image_raw': _bytes_feature(image_raw)
            }))
            writer.write(example.SerializeToString())

    shuffle(pipeline, _process_examples, output_name)


def shuffle_fashion_mnist(output_name):
    n_megabatchs = 5
    general_config = create_config(n_megabatchs)
    paths = get_paths_from_dataset(const.DATA_FASHION_MNIST)
    pipeline = FashionMnistData(general_config, paths[0], paths[1])

    def _process_examples(images, labels, writer):
        rows, cols = 28, 28
        for index in range(len(images)):
            sys.stdout.write(f"\rProcessing sample {index + 1}")
            sys.stdout.flush()
            image_raw = images[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'label': _int64_feature(int(labels[index])),
                'image_raw': _bytes_feature(image_raw)
            }))
            writer.write(example.SerializeToString())

    shuffle(pipeline, _process_examples, output_name)


def shuffle_cifar(output_name):
    n_megabatchs = 5
    general_config = create_config(n_megabatchs)
    paths = get_paths_from_dataset(const.DATA_CIFAR_10)
    pipeline = CifarData(general_config, paths[0], paths[1])

    def _process_examples(images, labels, writer):
        for index in range(len(images)):
            sys.stdout.write(f"\rProcessing sample {index + 1}")
            sys.stdout.flush()
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image': _bytes_feature(images[index].tobytes()),
                    'label': _int64_feature(labels[index])
                }))
            writer.write(example.SerializeToString())

    shuffle(pipeline, _process_examples, output_name)


def shuffle(pipeline, _process_examples, output_name):
    """
    Shuffles a dataset training data (only for TFRecords)
    :param pipeline: a Data pipeline
    :param _process_examples: a function that process examples, it is expected to receive 3 parameters: images, labels
    and writer, in that order.
    :param output_name: the base name of output files. E.g. if output_name = '..\batch_', then the output files are
    going to be: '..\batch_1', '..\batch_2', and so on
    :return: None
    """
    sess = tf.InteractiveSession()
    for i in range(len(pipeline.general_config.train_configurations)):
        pipeline.change_dataset_part(i)
        training_iterator, data_x, data_y = pipeline.build_train_data_tensor()
        sess.run(training_iterator.initializer)
        filename = "{}{}.tfrecords".format(output_name, i + 6)

        with tf.python_io.TFRecordWriter(filename) as writer:
            while True:
                try:
                    image_batch, target_batch = sess.run([data_x, data_y])
                    _process_examples(image_batch, target_batch, writer)
                    # Process samples
                except OutOfRangeError:
                    break


if __name__ == '__main__':
    shuffle_fashion_mnist("train-")
