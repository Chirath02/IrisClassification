import pandas as pd
import tensorflow as tf

TRAIN_DATA_PATH = 'data_set/iris_training.csv'
TEST_DATA_PATH = 'data_set/iris_test.csv'

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Sentosa', 'Versicolor', 'Virginica']


def load_data(label_name='Species'):
    """Parses the csv file in TRAIN_URL and TEST_URL."""

    # Parse the local CSV file.
    train = pd.read_csv(filepath_or_buffer=TRAIN_DATA_PATH,
                        names=CSV_COLUMN_NAMES,  # list of column names
                        header=0  # ignore the first row of the CSV file.
                       )
    train_features, train_label = train, train.pop(label_name)

    test = pd.read_csv(filepath_or_buffer=TEST_DATA_PATH,
                       names=CSV_COLUMN_NAMES,
                       header=0)

    test_features, test_label = test, test.pop(label_name)

    return (train_features, train_label), (test_features, test_label)
