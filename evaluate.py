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

    opts, args = getopt.getopt(argv, "hi:o:", ["animal_path=", "machine_path=", "binary_path="])

    animal_path = ""
    machine_path = ""
    binary_path = ""

    for opt, arg in opts:
        if opt in ("-a", "--animal_path"):
            animal_path = arg
        if opt in ("-m", "--machine_path"):
            machine_path = arg
        if opt in ("-b", "--binary_path"):
            binary_path = arg

    # machine = 0, animal = 1 for binary
    animal_labels = [2, 4, 5, 6, 7, 8]
    machine_labels = [1, 3, 9, 10]

    binary_model = get_model(binary_path,1)
    animal_model = get_model(animal_path,6)
    machine_model = get_model(machine_path,4)


    IMG_HEIGHT = 96
    IMG_WIDTH = 96

    binary_path = "data/stl10_binary/"

    # train_images = read_all_images(binary_path + "train_X.bin")
    # train_labels = read_labels(binary_path + "train_y.bin")

    test_images = read_all_images(binary_path + "test_X.bin")
    test_labels = read_labels(binary_path + "test_y.bin")

    animal_labels = [2, 4, 5, 6, 7, 8]
    machine_labels = [1, 3, 9, 10]
    binary_labels = [0, 1]
    # converted_labels = convert_labels(test_labels)


    # for i in range(len(test_images)):
        # return

    test_binary=True
    test_machine=True
    test_animal=True
    test_moe=True

    # if test_binary:
    #     binary_logits = binary_model.predict(test_images)
    #     binary_preds = tf.round(tf.nn.sigmoid(binary_logits))
    #     correct = 0
    #     total = 0
    #     binary_labels, binary_images = convert_labels(test_labels, test_images, "binary")
    #     for i in range(len(binary_images)):
    #         prediction = int(binary_preds[i])
    #         if prediction == binary_labels[i]:
    #             correct += 1
    #         total += 1
    #     print("Binary reported accuracy: {:5.2f}%".format(100 * correct/total))
    #     loss, acc = binary_model.evaluate(test_images, binary_labels , verbose=2)
    #     print("Binary true accuracy: {:5.2f}%".format(100 * acc))


    def test_model(name, model):
        correct = 0
        total = 0
        converted_class_labels, class_images = convert_labels(test_labels, test_images, name)

        if name == "binary":
            for i in range(len(converted_class_labels)):
                if converted_class_labels[i] != int(test_labels[i] in animal_labels):
                    print("Error in conversion for binary classification - exiting")
                    exit()
        class_logits = model.predict(class_images)

        if name != "binary":
            class_preds = np.argmax(tf.nn.softmax(class_logits), axis=1)
        else:
            # print(class_logits, class_logits.shape)
            class_preds = tf.round(tf.nn.sigmoid(class_logits))
        class_preds = tf.reshape(class_preds, [len(class_preds)])
        # class_preds = np.argmax(class_preds, axis=1)
        for i in range(len(class_images)):
            prediction = int(class_preds[i])
            total += 1
            if prediction == converted_class_labels[i]:
                correct += 1
        print(name + " reported accuracy: {:5.2f}%".format(100 * correct / total))
        loss, acc = model.evaluate(class_images, converted_class_labels, verbose=2)
        print(name + " true accuracy: {:5.2f}%".format(100 * acc))

    if test_binary:
        test_model("binary", binary_model)

    if test_machine:
        test_model("machine", machine_model)

    if test_animal:
        test_model("animal", animal_model)

    if test_moe:
        binary_logits = binary_model.predict(test_images)
        binary_preds = tf.round(tf.nn.sigmoid(binary_logits))
        machine_logits = machine_model.predict(test_images)
        machine_preds = tf.nn.softmax(machine_logits)
        animal_logits = animal_model.predict(test_images)
        animal_preds = tf.nn.softmax(animal_logits)
        correct = 0
        total = 0
        for i in range(len(test_images)):
            prediction = -1
            binary_prediction = int(binary_preds[i])
            if binary_prediction == 0:
                prediction = machine_labels[np.argmax(machine_preds[i])]
            else:
                prediction = animal_labels[np.argmax(animal_preds[i])]
            if prediction == test_labels[i]:
                correct += 1
            total += 1

        print("MoE reported accuracy: {:5.2f}%".format(100 * correct / total))


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