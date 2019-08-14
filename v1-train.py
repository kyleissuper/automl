import numpy as np

from autokeras import TextClassifier
from autokeras.utils import read_csv_file


def convert_labels_to_one_hot(labels, num_labels):
    labels = [int(label) for label in labels]
    one_hot = np.zeros((len(labels), num_labels))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


if __name__ == "__main__":
    file_path = "data/tmp_dataset.csv"
    x_train, y_train = read_csv_file(file_path)
    x_test, y_test = read_csv_file(file_path)

    y_train = convert_labels_to_one_hot(y_train, num_labels=5)
    y_test = convert_labels_to_one_hot(y_test, num_labels=5)

    clf = TextClassifier(verbose=True)
    clf.output_model_file = "data/v1.h5"
    clf.fit(x=x_train, y=y_train, time_limit=60)

    print("Classification accuracy is: ", 100 * clf.evaluate(
        x_test,
        y_test
        ), "%")
