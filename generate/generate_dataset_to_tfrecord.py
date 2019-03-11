"""Convert Dataset to local TFRecords"""

import argparse
import os
import random
import sys
from enum import Enum

import tensorflow as tf
import numpy
from tensorflow.examples.tutorials.mnist import input_data
from six.moves import cPickle as pickle
from six.moves import xrange  # pylint: disable=redefined-builtin


class MODES(Enum):
    BASIC = 0
    """
    This mode is for a balanced distribution of the samples over every megabatch
    """
    UNBALANCED = 1
    """
    This mode is for creating unbalanced increments of data, where each class is unevenly distributed across many
    megabatches
    """
    EXCLUSSIVE_CLASSES = 2
    """
    This mode is for creating a distribution of data where each class is present in **ONE and ONLY ONE** megabatch, but
    classes are distributed randomly across all the megabatches
    """
    EXCLUSSIVE_ORDERED_CLASSES = 3
    """
    This mode is for creating a distribution of data where each class is present in **ONE and ONLY ONE** megabatch, but
    classes are distributed ordered by label across all the megabatches
    """


def _data_path(data_directory: str, name: str) -> str:
    """Construct a full path to a TFRecord file to be stored in the 
    data_directory. Will also ensure the data directory exists
    
    Args:
        data_directory: The directory where the records will be stored
        name:           The name of the TFRecord
    
    Returns:
        The full path to the TFRecord file
    """
    if not os.path.isdir(data_directory):
        os.makedirs(data_directory)

    return os.path.join(data_directory, f'{name}.tfrecords')


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


def read_pickle_from_file(filename):
    with tf.gfile.Open(filename, 'rb') as f:
        if sys.version_info >= (3, 0):
            data_dict = pickle.load(f, encoding='bytes')
        else:
            data_dict = pickle.load(f)
    return data_dict


def mnist_convert_to(data_set, name: str, data_directory: str, num_shards: int = 1, mode: MODES = MODES.BASIC):
    """Convert the dataset into TFRecords on disk
    
    Args:
        data_set:       The MNIST data set to convert
        name:           The name of the data set
        data_directory: The directory where records will be stored
        num_shards:     The number of files on disk to separate records into
        mode:           The mode of generation of the dataset
    """
    print(f'Processing {name} data')
    images, labels = data_set.images, data_set.labels
    count = count_samples_by_label(labels)
    num_examples = int(numpy.sum(count))
    images_aux, labels_aux = create_aux_arrays(images, labels)

    k = 0
    labels.flags.writeable = True
    images = numpy.ndarray([num_examples, 28, 28, 1], dtype=numpy.float32)
    _, rows, cols, depth = images.shape

    distribution = create_distribution(count, num_shards, mode)

    # images = numpy.ndarray([num_examples, 28, 28], dtype=numpy.uint8)
    # Traverse the N megabatches distribution
    for n in distribution:
        # Traverse the M classes
        for label in range(0, len(n)):
            # Pops the number of desired samples
            for z in range(0, n[label]):
                aux = images_aux[label].pop()
                images[k] = aux
                labels[k] = labels_aux[label].pop()
                k += 1

    def _process_examples(start_idx: int, end_index: int, filename: str):
        with tf.python_io.TFRecordWriter(filename) as writer:
            for index in range(start_idx, end_index):
                sys.stdout.write(f"\rProcessing sample {index + 1} of {num_examples}")
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

    create_shards(_process_examples, distribution, name, data_directory, num_shards)


def fashion_mnist_convert_to(data_set, name: str, data_directory: str, num_shards: int = 1, mode: MODES = MODES.BASIC):
    """Convert the dataset into TFRecords on disk

    Args:
        data_set:       The FASHION-MNIST data set to convert
        name:           The name of the data set
        data_directory: The directory where records will be stored
        num_shards:     The number of files on disk to separate records into
        mode:           The mode of generation of the dataset
    """
    print(f'Processing {name} data')
    (images, labels) = data_set
    count = count_samples_by_label(labels)
    num_examples = int(numpy.sum(count))
    images_aux, labels_aux = create_aux_arrays(images, labels)

    k = 0
    labels.flags.writeable = True
    images = numpy.ndarray([num_examples, 28, 28], dtype=numpy.uint8)
    _, rows, cols = images.shape

    distribution = create_distribution(count, num_shards, mode)

    # images = numpy.ndarray([num_examples, 28, 28], dtype=numpy.uint8)
    # Traverse the N megabatches distribution
    for n in distribution:
        # Traverse the M classes
        for label in range(0, len(n)):
            # Pops the number of desired samples
            for z in range(0, n[label]):
                aux = images_aux[label].pop()
                images[k] = aux
                labels[k] = labels_aux[label].pop()
                k += 1

    def _process_examples(start_idx: int, end_index: int, filename: str):
        with tf.python_io.TFRecordWriter(filename) as writer:
            for index in range(start_idx, end_index):
                sys.stdout.write(f"\rProcessing sample {index + 1} of {num_examples}")
                sys.stdout.flush()

                image_raw = images[index].tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(rows),
                    'width': _int64_feature(cols),
                    'label': _int64_feature(int(labels[index])),
                    'image_raw': _bytes_feature(image_raw)
                }))
                writer.write(example.SerializeToString())

    create_shards(_process_examples, distribution, name, data_directory, num_shards)


def cifar_convert_to(data_set, name: str, data_directory: str, num_shards: int = 1, mode: MODES = MODES.BASIC):
    """Convert the dataset into TFRecords on disk

    Args:
        data_set:       The CIFAR data set to convert
        name:           The name of the data set
        data_directory: The directory where records will be stored
        num_shards:     The number of files on disk to separate records into
        mode:           The mode of generation of the dataset
    """
    print(f'Processing {name} data')
    (images, labels) = data_set
    count = count_samples_by_label(labels)
    num_examples = int(numpy.sum(count))
    images_aux, labels_aux = create_aux_arrays(images, labels)

    k = 0
    labels.flags.writeable = True
    images = numpy.ndarray([num_examples, 3072], dtype=numpy.uint8)

    distribution = create_distribution(count, num_shards, mode)

    # images = numpy.ndarray([num_examples, 28, 28], dtype=numpy.uint8)
    # Traverse the N megabatches distribution
    for n in distribution:
        # Traverse the M classes
        for label in range(0, len(n)):
            # Pops the number of desired samples
            for z in range(0, n[label]):
                aux = images_aux[label].pop()
                images[k] = aux
                labels[k] = labels_aux[label].pop()
                k += 1

    def _process_examples(start_idx: int, end_index: int, filename: str):
        with tf.python_io.TFRecordWriter(filename) as writer:
            for index in range(start_idx, end_index):
                sys.stdout.write(f"\rProcessing sample {index + 1} of {num_examples}")
                sys.stdout.flush()

                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image': _bytes_feature(images[index].tobytes()),
                        'label': _int64_feature(labels[index])
                    }))
                writer.write(example.SerializeToString())

    create_shards(_process_examples, distribution, name, data_directory, num_shards)


def create_aux_arrays(images, labels):
    """
    Creates the images and labels aux arrays
    :param images: the images array
    :param labels: the labels array
    :return: two 2D-arrays of images and labels (respectively) that distribute samples and labels by class
    """
    # Creates aux lists
    images_aux = list()
    labels_aux = list()

    # TODO: allow to define number of categories
    for i in range(0, 10):
        images_aux.append(list())
        labels_aux.append(list())

    # Fills the lists. Each list is associated with a class
    for i, label in enumerate(labels):
        label = int(label)
        images_aux[label].append(images[i])
        labels_aux[label].append(label)

    return images_aux, labels_aux


def create_distribution(count, num_shards, mode):
    """
    Creates the distribution array
    :param count: the count-per-class array
    :param num_shards: number of megabatches
    :param mode: The mode of generation of the dataset
    :return: a 2D-array with the distribution
    """
    if mode == MODES.BASIC:
        distribution = prepare_basic_distribution(count, num_shards)
    elif mode == MODES.UNBALANCED:
        distribution = prepare_unbalanced_distribution(count, 0.2, 0.35, num_shards)
    elif mode == MODES.EXCLUSSIVE_CLASSES:
        distribution = prepare_exc_class_distribution(count, num_shards)
    elif mode == MODES.EXCLUSSIVE_ORDERED_CLASSES:
        distribution = prepare_exc_ord_class_distribution(count, num_shards)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return distribution


def create_shards(process_examples, distribution, name, data_directory, num_shards):
    """
    This function creates the megabatches according to the distribution previously defined
    :param process_examples: a function to process samples and write them in disk
    :param distribution: the distribution 2D-array
    :param data_directory: The directory where records will be stored
    :param name: The name of the data set
    :param num_shards: the number of megabatches
    :return: None
    """
    num_examples = numpy.sum(distribution)
    if num_shards == 1:
        process_examples(0, num_examples, _data_path(data_directory, name))
    else:
        total_examples = num_examples
        # samples_per_shard = total_examples // num_shards
        samples_per_shard = numpy.sum(distribution, axis=1)
        start_index = 0

        for shard, n_samples in enumerate(samples_per_shard):
            end_index = start_index + n_samples
            process_examples(start_index, end_index, _data_path(data_directory, f'{name}-{shard + 1}'))
            start_index = end_index


def count_samples_by_label(labels):
    """
    Counts the samples by label
    :param data_set: the dataset that is going to be counted
    :return: a list where list[label] is equal to the number of instances of images with class label
    """
    count = []

    ind = numpy.argsort(labels)
    labels = numpy.take(labels, ind)
    for i in labels:
        i = int(i)
        if i < len(count):
            count[i] += 1
        else:
            count.append(1)
    return count


def prepare_basic_distribution(count_by_label, number_of_megabatchs):
    print("PREPARING DISTRIBUTION FOR MODE: BASIC...")
    distribution = [[0 for _ in range(len(count_by_label))] for _ in range(number_of_megabatchs)]
    number_of_categories = len(count_by_label)
    print("INFO: number of categories: ", number_of_categories)
    for i, count in enumerate(count_by_label):
        set_basic_distribution_for_label(distribution, count, i)

    print("DATASET DISTRIBUTION:")
    print(distribution)
    return distribution


def set_basic_distribution_for_label(distribution, count, label_index):
    current_count = count
    percentage = 1 / len(distribution)  # Balanced Class

    for megabatch in range(len(distribution)):
        excess = calculate_excess(count, current_count, percentage)
        current_count -= excess
        distribution[megabatch][label_index] = excess
    distribute_excess_for_label(distribution, current_count, label_index, min)


def prepare_unbalanced_distribution(count_by_label, percent_unbalanced, max_percent_unbalanced, number_of_megabatchs):
    print("PREPARING DISTRIBUTION FOR MODE: UNBALANCED...")
    distribution = [[0 for _ in range(len(count_by_label))] for _ in range(number_of_megabatchs)]
    number_of_categories = len(count_by_label)
    print("INFO: number of categories: ", number_of_categories)
    num_umbalacend_cantegories = int(number_of_categories * percent_unbalanced)
    print("INFO: number of umbalanced categories: ", num_umbalacend_cantegories)
    # TODO: randomize list of categories
    for i, count in enumerate(count_by_label):
        if i < num_umbalacend_cantegories:
            # UNBAlANCED CATEGORIES
            random_parameters = random.sample(range(0, number_of_megabatchs), 2)
            set_unbalanced_distribution_for_label(distribution, count, i, max_percent_unbalanced, random_parameters[0],
                                                  random_parameters[1])
        else:
            random_parameters = random.sample(range(0, number_of_megabatchs), 1)
            set_unbalanced_distribution_for_label(distribution, count, i, max_percent_unbalanced, random_parameters[0])
    print("DATASET DISTRIBUTION:")
    print(distribution)
    return distribution


def set_unbalanced_distribution_for_label(distribution, count, label_index, max_percent_unbalanced, min, max=None):
    current_count = count
    if max is not None:
        percentage = (1 - max_percent_unbalanced) / (len(distribution) - 2)  # Unbalanced class
    else:
        percentage = 1 / (len(distribution) - 1)  # Balanced Class

    for megabatch in range(len(distribution)):
        if megabatch == max:
            excess = calculate_excess(count, current_count, max_percent_unbalanced)
        elif megabatch == min:
            excess = 0
        else:
            excess = calculate_excess(count, current_count, percentage)
        current_count -= excess
        distribution[megabatch][label_index] = excess
    distribute_excess_for_label(distribution, current_count, label_index, min)


def prepare_exc_class_distribution(count_by_label, number_of_megabatchs):
    print("PREPARING DISTRIBUTION FOR MODE: EXCLUSSIVE_CLASS...")
    number_of_categories = len(count_by_label)
    distribution = [[0 for _ in range(number_of_categories)] for _ in range(number_of_megabatchs)]
    number_of_categories = len(count_by_label)
    categories = [i for i in range(number_of_categories)]
    print("INFO: number of categories: ", number_of_categories)
    current_megabatch = 0
    while len(categories) > 0:
        random_category = random.randint(0, len(categories) - 1)
        random_category = categories.pop(random_category)
        distribution[current_megabatch][random_category] = count_by_label[random_category]
        current_megabatch = current_megabatch + 1 if current_megabatch < number_of_megabatchs - 1 else 0

    print("DATASET DISTRIBUTION:")
    print(distribution)
    return distribution


def prepare_exc_ord_class_distribution(count_by_label, number_of_megabatchs):
    print("PREPARING DISTRIBUTION FOR MODE: EXCLUSSIVE_ORDERED_CLASS...")
    number_of_categories = len(count_by_label)
    distribution = [[0 for _ in range(number_of_categories)] for _ in range(number_of_megabatchs)]
    categories = [i for i in range(number_of_categories)]
    categories_per_batch = [int(number_of_categories/number_of_megabatchs) for _ in range(number_of_megabatchs)]
    excess = number_of_categories - int(numpy.sum(categories_per_batch))
    for i in range(excess):
        categories_per_batch[i] += 1
    print("INFO: number of categories: ", number_of_categories)
    current_megabatch = 0
    count = 0
    while len(categories) > 0:
        count += 1
        category = categories.pop(0)
        distribution[current_megabatch][category] = count_by_label[category]
        if count >= categories_per_batch[current_megabatch]:
            current_megabatch = current_megabatch + 1
            count = 0

    print("DATASET DISTRIBUTION:")
    print(distribution)
    return distribution


def distribute_excess_for_label(distribution, excess_count, label_index, min):
    """
    Distributes excess samples that weren't allocated in the previous process (allocation of number of samples by
    percentages). If there is no excess (excess_count=0) then the distribution remains intact
    """
    for megabatch in range(len(distribution)):
        if excess_count == 0:
            break
        if megabatch != min:
            distribution[megabatch][label_index] += 1
            excess_count -= 1


def calculate_excess(original_count, current_count, percentage):
    excess = int(percentage * original_count)
    return excess if current_count > excess else current_count


def mnist_convert_to_tf_record(data_directory: str, mode: MODES):
    """Convert the MNIST Dataset to TFRecord formats
    
    Args:
        data_directory: The directory where the TFRecord files should be stored
        mode: an integer representing the mode in which the data must be generated. These modes are stored
            in MODES enum
    """

    mnist = input_data.read_data_sets(
        "../../datasets/MNIST_data/",
        reshape=False
    )
    # convert_to(mnist.validation, 'validation', data_directory)
    mnist_convert_to(mnist.train, 'train', data_directory, num_shards=5, mode=mode)
    # convert_to(mnist.test, 'test', data_directory)


def fashion_mnist_convert_to_tf_record(data_directory: str, mode: MODES):
    """Convert the MNIST Dataset to TFRecord formats

    Args:
        data_directory: The directory where the TFRecord files should be stored
        mode: an integer representing the mode in which the data must be generated. These modes are stored
            in MODES enum
    """

    mnist = tf.keras.datasets.fashion_mnist.load_data()
    # convert_to(mnist.validation, 'validation', data_directory)
    fashion_mnist_convert_to(mnist[0], 'train', data_directory, num_shards=5, mode=mode)
    # convert_to(mnist[1], 'test', data_directory)


def cifar_convert_to_tf_record(data_directory: str, mode: MODES):
    """Convert the TF MNIST Dataset to TFRecord formats

    Args:
        data_directory: The directory where the TFRecord files should be stored
        mode: an integer representing the mode in which the data must be generated. These modes are stored
            in MODES enum
    """
    CIFAR_LOCAL_FOLDER = 'cifar-10-batches-py'
    file_names = {}
    file_names['train'] = ['data_batch_%d' % i for i in xrange(1, 6)]
    file_names['validation'] = ['data_batch_5']
    file_names['eval'] = 'test_batch'

    images = numpy.empty((0, 3072))
    labels = numpy.array([], dtype=numpy.int64)

    for direct in file_names['train']:
        data_dict = read_pickle_from_file(CIFAR_LOCAL_FOLDER + "/" + direct)
        images = numpy.append(images, data_dict[b'data'], axis=0)
        labels = numpy.append(labels, data_dict[b'labels'], axis=0)

    labels = numpy.asarray(labels)
    print(labels.shape)
    cifar_convert_to((images, labels), 'data_batch_', data_directory, num_shards=5, mode=mode)
    # convert_to(mnist.test, 'test', data_directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data-directory',
        default='../../datasets/EXTRA/',
        help='Directory where TFRecords will be stored')

    args = parser.parse_args()
    cifar_convert_to_tf_record(os.path.expanduser(args.data_directory), MODES.EXCLUSSIVE_ORDERED_CLASSES)
    print()
