"""
This module performs tests over a set of directories corresponding to a dataset divided in batches. The purpose of the
tests is to check the balancing of classes over multiple mega-batches, so that if the mega-batches are used in
incremental learning it'll easy to know the distribution of the data. The dataset must have a structure such that each
directory corresponds to a class, and each mega-batch is stored in one directory. The mega-batches must have the same
structure. The following structure is expected:

mega-batch-1/class-1/*
mega-batch-1/class-2/*
...
mega-batch-n/class-1/*
mega-batch-n/class-2/*

It calculates the following statistics:
1. Number of samples in each class for each mega-batch
3. Number of samples repeated in multiple mega-batches (Note: this is done assuming that samples with the same name
 within a class are equal samples)

It also can report a summary of findings:
1. Number of classes in each mega-batch
2. Total number of samples in each mega-batch
3. Differences in number of samples in each class across all the mega-batches
"""
import glob
import os
import numpy


def test_all(dirs: [str]):
    """
    Performs a number of tests over directories of data
    :param dirs: a list of directory paths
    :return: None
    """
    labels = set()
    paths = []
    repeated = []

    # Variables for statistics
    n_classes = numpy.zeros(len(dirs), dtype=int)
    n_samples = numpy.zeros(len(dirs), dtype=int)

    # Gets all the classes across all mega-batches
    for i in dirs:
        paths.append(i)
        labels.update(os.listdir(i))

    for label in labels:
        temp_files = set()
        local_sum = 0

        for i, path in enumerate(paths):
            full_path = os.path.join(path, label)

            if os.path.isdir(full_path):
                # The mega-batch has the class
                files = glob.glob(full_path + "/*")
                temp_files.update([os.path.split(files[x])[1] for x in range(len(files))])
                n_s = len(files)
                n_samples[i] += n_s
                if n_s > 0:
                    n_classes[i] += 1
                local_sum += n_s
                print("Mega-batch {} - Class '{}': {} samples".format(i, label, n_s))
            else:
                # The mega-batch DOESN'T have the class
                print("Mega-batch {} - Class '{}': NOT FOUND".format(i, label))

        if not len(temp_files) == local_sum:
            # There are repeated elements
            repeated.append("Class {} has {} elements repeated.".format(label, (local_sum - len(temp_files))))

        print("------------------------------------------------------")

    # Summary reports
    print("\n\n\n\n------------------------------------------------------")
    for i in range(len(n_classes)):
        print("Mega-batch {}: {} classes, {} samples.".format(i, n_classes[i], n_samples[i]))

    print("------------------------------------------------------")
    print("Total number of classes over all the mega-batches: {}".format(len(labels)))
    print("Total number of samples over all the mega-batches: {}".format(numpy.sum(n_samples, dtype=int)))
    print("------------------------------------------------------")

    if not repeated:
        print("There aren't any repeated elements in any of the classes")

    for i in repeated:
        print(i)


def main():
    root = "C:/Users/camil/Documents/Camilo/Trabajo de grado/datasets/101_ObjectCategories_UNBALANCED/"
    # root = "C:/Users/camil/Documents/Camilo/Trabajo de grado/datasets/tiny-imagenet-200/"
    paths = [(root+"train/Lote{}".format(x)) for x in range(0, 5)]
    # paths = [root+"val"]
    test_all(paths)


if __name__ == '__main__':
    main()
