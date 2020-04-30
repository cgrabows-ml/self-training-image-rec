import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data.stl10_input import read_labels, read_all_images

import matplotlib.pyplot as plt
import numpy as np
import math
import os

import sys, getopt

from util import convert_labels

def main(argv):

    opts, args = getopt.getopt(argv, "hi:o:", ["a=", "b=", "m=", "animal_epoch=", "machine_epoch=", "binary_epoch="])

    animal_path = ""
    machine_path = ""
    training_path = "training/"

    for opt, arg in opts:
        if opt in ("-a", "--animal_epoch"):
            animal_path = training_path + "disagreement_animal-cp-{:04d}.ckpt".format(int(arg))
        if opt in ("-m", "--machine_epoch"):
            machine_path = training_path + "disagreement_machine-cp-{:04d}.ckpt".format(int(arg))

        # if opt in ("-b", "--binary_epoch"):
        #     binary_path = training_path + "binary-cp-{:04d}.ckpt".format(int(arg))

    # machine = 0, animal = 1 for binary
    animal_labels = [2, 4, 5, 6, 7, 8]
    machine_labels = [1, 3, 9, 10]

    print("Path information:", animal_path, machine_path)
    # exit()

    animal_model = get_model(animal_path,7)
    machine_model = get_model(machine_path,5)


    IMG_HEIGHT = 96
    IMG_WIDTH = 96

    binary_path = "data/stl10_binary/"

    # train_images = read_all_images(binary_path + "train_X.bin")
    # train_labels = read_labels(binary_path + "train_y.bin")

    test_images = read_all_images(binary_path + "test_X.bin")
    test_labels = read_labels(binary_path + "test_y.bin")

    animal_labels = [2, 4, 5, 6, 7, 8]
    machine_labels = [1, 3, 9, 10]

    test_machine=True
    test_animal=True
    test_disagreement=True


    def test_model(name, model):
        correct = 0
        total = 0
        converted_class_labels, class_images = convert_labels(test_labels, test_images, name)


        class_logits = model.predict(class_images)

        class_preds = np.argmax(tf.nn.softmax(class_logits), axis=1)
        # class_preds = tf.round(tf.nn.sigmoid(class_logits))
        class_preds = tf.reshape(class_preds, [len(class_preds)])
        for i in range(len(class_images)):
            prediction = int(class_preds[i])
            total += 1
            if prediction == converted_class_labels[i]:
                correct += 1
        print(name + " reported accuracy: {:5.2f}%".format(100 * correct / total))
        loss, acc = model.evaluate(class_images, converted_class_labels, verbose=2)
        print(name + " true accuracy: {:5.2f}%".format(100 * acc))


    if test_machine:
        test_model("disagreement_machine", machine_model)

    if test_animal:
        test_model("disagreement_animal", animal_model)

    if test_disagreement:
        animal_labels = [2, 4, 5, 6, 7, 8]
        machine_labels = [1, 3, 9, 10]
        animal_predictions = animal_model.predict(test_images)
        animal_predictions = tf.nn.softmax(animal_predictions)
        print("animal predictions shape: ", animal_predictions.shape)
        animal_class_preds = np.amax(animal_predictions[:, 0:6], axis=1)
        animal_other_preds = animal_predictions[:, 6]
        animal_class_labels = np.argmax(animal_predictions[:, 0:6], axis=1)

        machine_predictions = machine_model.predict(test_images)
        print("machine predictions shape: ", machine_predictions.shape)
        machine_class_preds = np.amax(machine_predictions[:, 0:4], axis=1)
        machine_other_preds = machine_predictions[:, 4]
        machine_class_labels = np.argmax(machine_predictions[:, 0:4], axis=1)

        model_preds = []
        for i in range(len(test_labels)):
            if animal_class_preds[i] * machine_other_preds[i] > \
                            animal_other_preds[i] * machine_class_preds[i]:
                model_preds.append(animal_labels[animal_class_labels[i]])
            else:
                model_preds.append(machine_labels[machine_class_labels[i]])

        valid_accuracy = np.sum(model_preds == test_labels) / len(test_labels)
        print("validation accuracy: ", valid_accuracy)

        print("Disagreement reported accuracy: {:5.2f}%".format(100 * valid_accuracy))


# def convert_labels_old(labels):
#     animal_labels = [2, 4, 5, 6, 7, 8]
#     machine_labels = [1, 3, 9, 10]
#     new_labels = []
#     # print(labels)
#     for label in labels:
#         # print(label)
#         if label in animal_labels:
#             new_labels.append(1)
#         elif label in machine_labels:
#             new_labels.append(0)
#         else:
#             print("Error in assigning labels")
#     return np.array(new_labels)


def get_model(weights_path, num_classes):
    IMG_HEIGHT = 96
    IMG_WIDTH = 96

    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu',
               input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(num_classes)
    ])

    if num_classes == 1:
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])
    else:
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

    model.load_weights(weights_path)
    return model


if __name__ == "__main__":
    main(sys.argv[1:])