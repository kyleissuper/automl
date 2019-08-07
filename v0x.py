from autokeras.utils import read_csv_file

X, y = read_csv_file("tmp_dataset.csv")
print(X, y)
