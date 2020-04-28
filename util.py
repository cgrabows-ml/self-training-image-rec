import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

import numpy as np


def convert_labels(labels, images, experiment):
    print(experiment)
    animal_labels = [2,4,5,6,7,8]
    machine_labels = [1,3,9,10]
    new_labels = []
    new_images = []

    if experiment!="default":
        for i in range(len(labels)):
            if labels[i] in animal_labels:
                if experiment == "animal":
                    new_labels.append(animal_labels.index(labels[i]))
                    new_images.append(images[i])
                elif experiment == "binary":
                    new_labels.append(1)
                    new_images.append(images[i])
                elif experiment == "disagreement_machine":
                    new_labels.append(len(machine_labels))
                    new_images.append(images[i])
                elif experiment == "disagreement_animal":
                    new_labels.append(animal_labels.index(labels[i]))
                    new_images.append(images[i])
            elif labels[i] in machine_labels:
                if experiment == "machine":
                    new_labels.append(machine_labels.index(labels[i]))
                    new_images.append(images[i])
                elif experiment == "binary":
                    new_labels.append(0)
                    new_images.append(images[i])
                elif experiment == "disagreement_animal":
                    new_labels.append(len(animal_labels))
                    new_images.append(images[i])
                elif experiment == "disagreement_machine":
                    new_labels.append(machine_labels.index(labels[i]))
                    new_images.append(images[i])
            else:
                print("Error in assigning labels - exiting")
                exit()
    elif experiment == "default":
        new_labels = labels
        new_images = images
    else:
        print("Invalid experiment - exiting")
    new_labels = np.array(new_labels)
    if experiment == "default":
        new_labels = new_labels-1

    return np.array(new_labels), np.array(new_images)


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