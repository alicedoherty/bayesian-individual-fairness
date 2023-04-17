# For installation of DeepBayes see https://stackoverflow.com/questions/23075397/python-how-to-edit-an-installed-package
import deepbayes.optimizers as optimizers
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from DNN_FGSM import DNN_FGSM
from BNN_FGSM import BNN_FGSM
from datetime import datetime
import random


def preprocess_data():
    # 1. Load UCI adult dataset
    features = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "salary"]

    data_path = "./data/adult.data"
    input_data = pd.read_csv(data_path, names=features,
                             sep=r'\s*,\s*', engine='python', na_values="?")

    # 2. Clean up dataset
    # Binary classification: 1 = >50K, 0 = <=50K
    y = (input_data['salary'] == '>50K').astype(int)

    # Features x contain categorical data, so use pandas.get_dummies to convert to one-hot encoding
    x = (input_data
         .drop(columns=['salary'])
         .fillna('Unknown')
         .pipe(pd.get_dummies, drop_first=True))

    # Normalise data
    x = x/np.max(x)

    # 3. Split data into training and test sets
    # 80% training, 20% test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y)

    # 4. Preprocess training and test data
    # Flattening data to 1D vector
    x_train = x_train.values.reshape(x_train.shape[0], -1)
    x_test = x_test.values.reshape(x_test.shape[0], -1)

    # Convert class vectors to binary class matrices using one-hot encoding
    num_classes = 2
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test


def train_DNN_model(x_train, y_train, model):
    # Printing the model to standard output
    model.summary()

    # Initialising some training parameters
    loss = keras.losses.CategoricalCrossentropy()
    optimizer = keras.optimizers.Adam()
    batch_size = 128
    epochs = 15     # See EarlyStopping callback below
    validation_split = 0.1
    # callbacks = [keras.callbacks.EarlyStopping(patience=2)]

    model.compile(loss=loss,
                  optimizer=optimizer, metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=epochs, validation_split=validation_split)

    return model


def train_BNN_model(x_train, y_train, x_test, y_test, model):
    # Printing the model to standard output
    model.summary()

    # Initialising some training parameters
    loss = keras.losses.CategoricalCrossentropy()
    # Deepbayes Adam optimizer doesn't seem to work properly
    # opt = optimizers.Adam()
    optimizer = optimizers.VariationalOnlineGuassNewton()
    batch_size = 128
    epochs = 15
    # validation_split = 0.1

    bayes_model = optimizer.compile(
        model, loss_fn=loss, batch_size=batch_size, epochs=epochs)

    # Why does it need x_test and y_test for training?
    bayes_model.train(x_train, y_train, x_test, y_test)

    return bayes_model


def get_adversarial_examples(model, test_data, epsilon, type):
    epsilons = np.full(100, epsilon)

    # Index 58 is the feature for gender (0 for Female, 1 for Male)
    epsilons[58] = 1.0
    adversarial_examples = np.ndarray(shape=(test_data.shape))

    for i in range(len(test_data)):
        if type == "DNN":
            adversarial = DNN_FGSM(
                model, test_data[i], keras.losses.categorical_crossentropy, epsilons)
        elif type == "BNN":
            adversarial = BNN_FGSM(
                model, test_data[i], keras.losses.categorical_crossentropy, epsilons)
        adversarial_examples[i] = adversarial

    return adversarial_examples


def get_fairness_score_basic(x_test_predictions, x_test_adversarial_predictions, type):
    classes = ["<=50K", ">50K"]
    count = 0

    for i in range(len(x_test_predictions)):
        if classes[np.argmax(x_test_predictions[i])] != classes[np.argmax(
                x_test_adversarial_predictions[i])]:
            count += 1

    basic_score = (1 - (count / len(x_test_predictions)))
    print(count, "/", len(x_test_predictions),
          "individuals classified differently after adversarial attack")
    print("Basic Fairness Score:", basic_score, "\n")

    return basic_score


def get_fairness_score(x_test_predictions, x_test_adversarial_predictions, type):
    differences = []

    if type == "DNN":
        for i in range(len(x_test_predictions)):
            difference = abs(x_test_predictions[i][0] -
                             x_test_adversarial_predictions[i][0])
            differences.append(difference)
    elif type == "BNN":
        for i in range(len(x_test_predictions)):
            difference = abs(x_test_predictions[i][0] -
                             x_test_adversarial_predictions[i][0]).numpy()
            differences.append(difference)

    max_diff = max(differences)
    min_diff = min(differences)
    avrg_diff = np.mean(differences)

    print("Maximum Difference:", max_diff)
    print("Minimum Difference:", min_diff)
    print("Average (Mean) Difference:", avrg_diff, "\n")

    return max_diff, min_diff, avrg_diff


def get_results(model, x_test, y_test, epsilon, type):
    # Get predictions for x_test data (without attack)
    x_test_predictions = model.predict(x_test)

    # Get numpy array of x_test data converted to adversarial examples
    x_test_adversarial = get_adversarial_examples(model, x_test, epsilon, type)

    # Get predictions for x_test_adversarial data (with attack)
    x_test_adversarial_predictions = model.predict(x_test_adversarial)

    # Get accuracy of model
    if type == "DNN":
        score = model.evaluate(x_test, y_test)
        accuracy = score[1]
    elif type == "BNN":
        test_acc = np.mean(np.argmax(x_test_predictions, axis=1)
                           == np.argmax(y_test, axis=1))
        accuracy = test_acc

    print(f"\n ❗️{type} RESULTS❗️")
    print("Accuracy:", accuracy, "\n")

    basic_score = get_fairness_score_basic(
        x_test_predictions, x_test_adversarial_predictions, type)

    max_diff, min_diff, avrg_diff = get_fairness_score(x_test_predictions,
                                                       x_test_adversarial_predictions, type)

    return basic_score, max_diff, min_diff, avrg_diff, accuracy


def main():
    x_train, x_test, y_train, y_test = preprocess_data()

    # Try 0.00, 0.05, 0.10, 0.15, 0.20
    epsilon = 0.20

    input_shape = x_train.shape[1]
    num_classes = y_train.shape[1]

    # Number of hidden layers in the model
    layers = [1, 2, 3, 4, 5]
    # Number of neurons per hidden layer in the model
    neurons = [64, 32, 16, 8, 4, 2]

    # Measurements we're recording during the trials
    measurements = ["DNNAccuracy", "BNNAccuracy", "DNNBasicScore", "BNNBasicScore",
                    "DNNMaxDifference", "BNNMaxDifference", "DNNMinDifference", "BNNMinDifference", "DNNMeanDifference", "BNNMeanDifference"]

    # Order of models tested: L1N64, L2N64,... , L5N64, L1N32, ..., L5N32, ..., L1N2, ..., L5N2
    # Where, L = number of hidden layers (1, 2, 3, 4, 5)
    # and, N = number of neurons per layer, i.e width (64, 32, 16, 8, 4, 2)
    # Order is because of how we want the data to be displayed in heatmap (i.e. first five datapoints for a measurement correspond to the top/first row of heatmap)
    # Heatmap Layout:
    # N64
    # N32
    # N16
    # N8
    # N4
    # N2
    #    L1 L2 L3 L4 L5

    df = pd.DataFrame(columns=measurements)
    # Number of neurons per layer (64, 32, 16, 8, 4, 2)
    for neuron_num in neurons:
        # Number of layers (1, 2, 3, 4, 5)
        for layer_num in layers:
            # Reason why model_DNN and model_BNN are defined separately (even though they're the same)
            # is to do with how Python passes values/objects through functions
            # http://scipy-lectures.org/intro/language/functions.html#passing-by-value

            model_DNN = keras.Sequential()
            model_DNN.add(keras.Input(shape=input_shape))

            model_BNN = keras.Sequential()
            model_BNN.add(keras.Input(shape=input_shape))

            for x in range(layer_num):
                model_DNN.add(keras.layers.Dense(
                    neuron_num, activation="relu"))
                model_BNN.add(keras.layers.Dense(
                    neuron_num, activation="relu"))

            model_DNN.add(keras.layers.Dense(
                num_classes, activation="softmax"))

            model_BNN.add(keras.layers.Dense(
                num_classes, activation="softmax"))

            trained_model_DNN = train_DNN_model(
                x_train, y_train, model_DNN)
            trained_model_BNN = train_BNN_model(
                x_train, y_train, x_test, y_test, model_BNN)

            basic_score_DNN, max_diff_DNN, min_diff_DNN, avrg_diff_DNN, accuracy_DNN = get_results(
                trained_model_DNN, x_test, y_test, epsilon, "DNN")

            basic_score_BNN, max_diff_BNN, min_diff_BNN, avrg_diff_BNN, accuracy_BNN = get_results(
                trained_model_BNN, x_test, y_test, epsilon, "BNN")

            # Dummy data for debugging
            # accuracy_DNN = random.random()
            # accuracy_BNN = random.random()
            # basic_score_DNN = random.random()
            # basic_score_BNN = random.random()
            # max_diff_DNN = random.random()
            # max_diff_BNN = random.random()
            # min_diff_DNN = random.random()
            # min_diff_BNN = random.random()
            # avrg_diff_DNN = random.random()
            # avrg_diff_BNN = random.random()

            new_row = pd.DataFrame([accuracy_DNN, accuracy_BNN, basic_score_DNN, basic_score_BNN, max_diff_DNN, max_diff_BNN,
                                    min_diff_DNN, min_diff_BNN, avrg_diff_DNN, avrg_diff_BNN], index=measurements, columns=[f"L{layer_num}N{neuron_num}"]).T
            df = pd.concat((df, new_row))

    # Pandas options to display all columns and rows
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    # pd.set_option('display.max_rows', None)
    # np.set_printoptions(linewidth=100000)

    f = open(f"./final/results/trial_{datetime.now()}_eps_{epsilon}.csv", 'a')
    print(df, file=f)


if __name__ == "__main__":
    main()
