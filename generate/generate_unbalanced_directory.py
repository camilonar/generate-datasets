import shutil, os
import numpy as np
import random

DEFAULT_PATH_BASE = "../../datasets/101_ObjectCategories/train"


def _copy_files(base_path, category, files, destination_path):
    if (os.path.isdir(destination_path) == False):
        os.makedirs(destination_path)
    for file in files:
        shutil.copy(base_path + "/" + category + "/" + file, destination_path)


class Category:
    def __init__(self, category_name, example_list, mega_batch_missing=None, mega_batch_max=None):
        self.category_name = category_name
        self.example_list = example_list
        self.total_examples = len(example_list)
        self.mega_batch_missing = mega_batch_missing
        self.mega_batch_max = mega_batch_max
        # self.print_category()

    def pop_n_examples(self, n):
        if n > 0:
            values = self.example_list[:n]
            del self.example_list[:n]
            return values
        else:
            return []

    def print_category(self):
        print("CATEGORY: ", self.category_name)
        print("     Megabatch missing: ", self.mega_batch_missing)
        print("     Megabatch max: ", self.mega_batch_max)


def prepare_categories(path_categories, percent_unbalanced, number_of_megabatchs):
    categories = []  # list of Categories class where save parameters for create megabatchs
    categories_list = os.listdir(path_categories)
    number_of_categories = len(categories_list)
    print("INFO: number of categories: ", number_of_categories)
    num_umbalacend_cantegories = int(number_of_categories * percent_unbalanced)
    print("INFO: number of umbalanced categories: ", num_umbalacend_cantegories)
    # TODO: randomize list of categories
    for i, category_name in enumerate(categories_list):
        example_list = os.listdir(path_categories + "/" + category_name)
        if (i < num_umbalacend_cantegories):
            # UNBAlANCED CATEGORIES
            random_parameters = random.sample(range(0, number_of_megabatchs), 2)
            categories.append(Category(category_name, example_list, random_parameters[0], random_parameters[1]))
        else:
            random_parameters = random.sample(range(0, number_of_megabatchs), 1)
            categories.append(Category(category_name, example_list, random_parameters[0]))
    return categories


def create_unbalanced_dataset(number_of_megabatchs, max_percent_unbalanced, categories, path_categories, save_path,
                              name_megabatch="Lote"):
    if max is not None:
        percentage = ((1 - max_percent_unbalanced) / (number_of_megabatchs - 2))  # Unbalanced class
    else:
        percentage = 1 / (number_of_megabatchs - 1)  # Balanced Class

    for i in range(number_of_megabatchs):
        for category in categories:
            current_count = len(category.example_list)
            if i == category.mega_batch_max:
                num_examples = calculate_excess(category.total_examples, current_count, max_percent_unbalanced)
            elif i == category.mega_batch_missing:
                num_examples = 0
            else:
                num_examples = calculate_excess(category.total_examples, current_count, percentage)
            list_examples = category.pop_n_examples(num_examples)
            destination_path = save_path + "/" + name_megabatch + str(i) + "/" + category.category_name
            _copy_files(path_categories, category.category_name, list_examples, destination_path)

    # Distribute excess
    for category in categories:
        excess_count = len(category.example_list)
        curr_megabatch = 0
        while excess_count > 0:
            if curr_megabatch != category.mega_batch_missing:
                excess_count -= 1
                list_examples = category.pop_n_examples(1)
                destination_path = save_path + "/" + name_megabatch + str(curr_megabatch) + "/" + category.category_name
                _copy_files(path_categories, category.category_name, list_examples, destination_path)
            curr_megabatch = curr_megabatch + 1 if curr_megabatch < number_of_megabatchs - 1 else 0


def create_exc_classes_ord_dataset(number_of_megabatchs, path_categories, save_path, name_megabatch="Lote"):
    categories = []  # list of Categories class where save parameters for create megabatchs
    categories_list = sorted(os.listdir(path_categories))
    number_of_categories = len(categories_list)
    print("INFO: number of categories: ", number_of_categories)
    current_megabatch = 0
    categories_per_batch = [int(number_of_categories/number_of_megabatchs) for _ in range(number_of_megabatchs)]
    excess = number_of_categories - int(np.sum(categories_per_batch))
    for i in range(excess):
        categories_per_batch[i] += 1
    # TODO: randomize list of categories
    count = 0
    while current_megabatch < number_of_megabatchs:
        count += 1
        category_name = categories_list[0]
        categories_list.pop(0)
        example_list = os.listdir(path_categories + "/" + category_name)
        categories.append(Category(category_name, example_list, mega_batch_max=current_megabatch))
        if count >= categories_per_batch[current_megabatch]:
            current_megabatch = current_megabatch + 1
            count = 0

    for i in range(number_of_megabatchs):
        for category in categories:
            destination_path = save_path + "/" + name_megabatch + str(i) + "/" + category.category_name
            if category.mega_batch_max == i:
                num_examples = int(category.total_examples)
                list_examples = category.pop_n_examples(num_examples)
                _copy_files(path_categories, category.category_name, list_examples, destination_path)
            elif not os.path.isdir(destination_path):
                os.makedirs(destination_path)


def create_exc_classes_rand_dataset(number_of_megabatchs, path_categories, save_path, name_megabatch="Lote"):
    categories = []  # list of Categories class where save parameters for create megabatchs
    categories_list = os.listdir(path_categories)
    number_of_categories = len(categories_list)
    print("INFO: number of categories: ", number_of_categories)
    categories_indices = [i for i in range(number_of_categories)]
    current_megabatch = 0
    # TODO: randomize list of categories
    while len(categories_indices) > 0:
        random_category = random.randint(0, len(categories_indices) - 1)
        random_category = categories_indices.pop(random_category)
        category_name = categories_list[random_category]
        example_list = os.listdir(path_categories + "/" + category_name)
        categories.append(Category(category_name, example_list, mega_batch_max=current_megabatch))
        current_megabatch = current_megabatch + 1 if current_megabatch < number_of_megabatchs - 1 else 0
    for i in range(number_of_megabatchs):
        for category in categories:
            destination_path = save_path + "/" + name_megabatch + str(i) + "/" + category.category_name
            if category.mega_batch_max == i:
                num_examples = int(category.total_examples)
                list_examples = category.pop_n_examples(num_examples)
                _copy_files(path_categories, category.category_name, list_examples, destination_path)
            elif not os.path.isdir(destination_path):
                os.makedirs(destination_path)


def calculate_excess(original_count, current_count, percentage):
    excess = int(percentage * original_count)
    return excess if current_count > excess else current_count


# categories = prepare_categories(DEFAULT_PATH_BASE, 0.3, 5)
# create_unbalanced_dataset(5, 0.35, categories, DEFAULT_PATH_BASE, '../../datasets/MNIST_BALANCED/train/')
# create_exc_classes_rand_dataset(5, DEFAULT_PATH_BASE, '../../datasets/MNIST_BALANCED/train/')
create_exc_classes_ord_dataset(5, DEFAULT_PATH_BASE, '../../datasets/EXTRA_CALTECH/train/')
