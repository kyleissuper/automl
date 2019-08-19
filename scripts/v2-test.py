import numpy as np

from autokeras import TextClassifier
from autokeras.utils import read_csv_file

import csv


def convert_labels_to_one_hot(labels, num_labels):
    labels = [int(label) for label in labels]
    one_hot = np.zeros((len(labels), num_labels))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

def convert_one_hot_to_labels(one_hots, num_labels):
    labels = []
    for one_hot in one_hots:
        for label in range(num_labels):
            if one_hot[label] == 1:
                labels.append(label)
                break
    return labels


if __name__ == "__main__":
    file_path = "../data/w_train_v3.csv"
    x_test, y_test = read_csv_file(file_path)
    y_test = convert_labels_to_one_hot(y_test, num_labels=3)

    clf = TextClassifier(verbose=True)
    clf.num_labels = 3
    clf.output_model_file = "../data/v2.h5"

    predictions = zip(
            x_test,
            convert_one_hot_to_labels(y_test, num_labels=3),
            clf.predict(x_test)
            )
    csvfile = "../data/w_test_output.csv"
    with open(csvfile, "w") as output:
        writer = csv.writer(output)
        for line in predictions:
            writer.writerow(line)
    

    # print("Classification accuracy is: ", 100 * clf.evaluate(
    #     x_test,
    #     y_test
    #     ), "%")
