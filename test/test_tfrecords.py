"""
This module performs tests over a set of TFRecords corresponding to a dataset divided in batches. The purpose of the
tests is to check the balancing of classes over multiple mega-batches, so that if the mega-batches are used in
incremental learning it'll easy to know the distribution of the data. The TFRecords must have the same structure, and
this structure has to be provided to the program when the test starts.

It calculates the following statistics:
1. Number of samples in each class for each mega-batch
3. Number of samples repeated in multiple mega-batches (Note: this is done assuming that samples with the same name
 within a class are equal samples)

It also can report a summary of findings:
1. Number of classes in each mega-batch
2. Total number of samples in a class across multiple mega-batches
3. Total number of samples in each mega-batch
4. Differences in number of samples in each class across all the mega-batches
"""
from collections import OrderedDict

import tensorflow as tf

# TODO: usar sets para guardar las samples
from tensorflow.python.framework.errors_impl import OutOfRangeError


def test_all(paths: [str], feature: {}, image_type, test_repeated=False):
    """
    Performs a number of tests over TFRecords of data.
    NOTE: this function currently only supports datasets composed of images that have an associated label
    :param paths: a list of paths to the TFRecords of a dataset
    :param feature: a dictionary with the structure of the data. For example:
        feature={
                    'image': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.int64)
                }
    The first value inserted in the dictionary must correspond to the image and the second one to the label, although
    there can be more fields and the keys don't need to be exactly 'image' and 'label' (like 'raw' instead of 'image')
    NOTE: it is suggested to use an OrderedDict to assure the order of the inserted keys
        :param image_type: the data type of the image (e.g. tf.float32, numpy.uint8)
    :param test_repeated: if True, then the test includes a validation to see if there are any repeated samples across
    multiple batches (for example, if image X is located in mega-batch 1 and in mega-batch 3). This test has the
    limitation that the whole dataset must fit into memory.
    :return: None
    """

    sess = tf.InteractiveSession()
    images_set = set()
    labels_count = {}
    count = 0

    for i, path in enumerate(paths):
        aux_labels, aux_set = test_one_mega_batch(i, path, sess, image_type, feature)
        if test_repeated:
            images_set = images_set.union(aux_set)

        # Loads local label counts into the full labels counts
        for label in aux_labels.keys():
            count += aux_labels[label]
            if label not in labels_count:
                labels_count[label] = aux_labels[label]
            else:
                labels_count[label] += aux_labels[label]

    # Summaries
    print("\n\n\n\n------------------------------------------------------")
    for k, v in labels_count.items():
        print(" Class '{}': {} total samples".format(k, v))

    print("------------------------------------------------------")
    if test_repeated:
        if count == len(images_set):
            print("There are NO repeated samples across multiple mega-batches")
        else:
            print("There are {} repeated samples  across multiple mega-batches".format(count - len(images_set)))

    print("------------------------------------------------------")
    print("Total number of classes over all the mega-batches: {}".format(len(labels_count)))
    print("Total number of samples over all the mega-batches: {}".format(count))
    print("------------------------------------------------------")
    sess.close()


def test_one_mega_batch(index: int, path: str, sess: tf.InteractiveSession, feature: {}, image_type):
    """
    Performs tests over one mega-batch.
    NOTE: this function currently only supports datasets composed of images that have an associated label
    :param index: the number of the mega-batch
    :param path: a path to one TFRecords file
    :param sess: the current session
    :param feature: a dictionary with the structure of the data. For example:
        feature={
                    'image': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.int64)
                }
    The first value inserted in the dictionary must correspond to the image and the second one to the label, although
    there can be more fields and the keys don't need to be exactly 'image' and 'label' (like 'raw' instead of 'image')
    NOTE: it is suggested to use an OrderedDict to assure the order of the inserted keys
    :param image_type: the data type of the image (e.g. tf.float32, numpy.uint8)
    :return: a tuple containing two values in the following order:
        -An ordered dict with the classes and number of samples of each class present in the mega-batch.
        E.g. {'c1':20, 'c2':30}
        -A set containing all the unique images of the mega-batch
    """
    labels_count = {}
    images_set = set()

    feature = OrderedDict(feature)
    keys = list(feature.keys())

    def parser(serialized_example):
        """Parses a single tf.Example into image and label tensors.
        :param serialized_example: a single sample from the TFRecord"""
        features = tf.parse_single_example(
            serialized_example,
            features=feature)
        image = tf.decode_raw(features[keys[0]], image_type)
        label = tf.cast(features[keys[1]], tf.int32)
        return image, label

    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(parser)
    iterator = dataset.make_one_shot_iterator()
    images_tensor, labels_tensor = iterator.get_next()

    count = 0
    from matplotlib import pyplot
    while True:
        try:
            image, label = sess.run([images_tensor, labels_tensor])
            if label not in labels_count:
                labels_count[label] = 1
            else:
                labels_count[label] += 1
            count += 1
            images_set.add(tuple(image))

        except OutOfRangeError:
            break

    # Summaries of a single mega-batch
    labels_count = OrderedDict(sorted(labels_count.items()))
    for k, v in labels_count.items():
        print("Mega-batch {} - Class '{}': {} samples".format(index, k, v))

    print("------------------------------------------------------")
    if count == len(images_set):
        print("There are NO repeated samples in mega-batch {}".format(index))
    else:
        print("There are {} repeated samples in mega-batch {}".format(count - len(images_set), index))

    print("------------------------------------------------------")
    print("Total number of classes in the mega-batch {}: {}".format(index, len(labels_count)))
    print("Total number of samples in the mega-batch {}: {}".format(index, count))
    print("------------------------------------------------------\n")
    return labels_count, images_set


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


def main():
    root = "C:/Users/camil/Documents/Camilo/Trabajo de grado/datasets/FASHION-MNIST_UNBALANCED/"
    # root = "C:/Users/camil/Documents/Camilo/Trabajo de grado/datasets/cifar10/"
    paths = [(root+"train-{}.tfrecords".format(x)) for x in range(1, 6)]
    paths.extend([root + "test.tfrecords"])
    features = {
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
    #    'depth': tf.FixedLenFeature([], tf.int64)
    }
    # features = {
    #     'image': tf.FixedLenFeature([], tf.string),
    #     'label': tf.FixedLenFeature([], tf.int64)
    # }
    test_all(paths, features, tf.float32, test_repeated=True)


if __name__ == '__main__':
    main()
