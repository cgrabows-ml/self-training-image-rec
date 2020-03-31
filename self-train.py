import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data.stl10_input import read_labels, read_all_images

import matplotlib.pyplot as plt
import numpy as np
import math

import sys, getopt

def main(argv):

    opts, args = getopt.getopt(argv, "hi:o:", ["binary_classification=", "subset=", "self_train=", "epochs=", "batch=", "threshold="])

    binary_classification = False
    self_train = False
    subset_size = 1
    epochs = 15
    batch_size = 1000
    threshold = .75
    name = "default"

    for opt, arg in opts:
        if opt == '-h':
            print('self-train.py --binary_classification=false --subset=1 --self_train=True --epochs=20 --batch=1000 --threshold=.75')
            sys.exit()
        elif opt in ("-st", "--self_train"):
            self_train= arg.lower in ["true", "True"]
        elif opt in ("-ss", "--subset"):
            subset_size = float(arg)
        elif opt in ("-bc", "--binary_classification"):
            binary_classification = bool(arg)
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-b", "--batch"):
            batch_size = int(arg)
        elif opt in ("-t", "--threshold"):
            threshold = float(arg)
        elif opt in ("-n", "--name"):
            name = float(arg)

    print(self_train)
    IMG_HEIGHT = 96
    IMG_WIDTH = 96

    binary_path = "data/stl10_binary/"

    train_images = read_all_images(binary_path + "train_x.bin")
    train_labels = read_labels(binary_path + "train_Y.bin")

    test_images = read_all_images(binary_path + "test_x.bin")
    test_labels = read_labels(binary_path + "test_Y.bin")


    def convert_labels(labels):
        animal_labels = [2,4,5,6,7,8]
        machine_labels = [1,3,9,10]
        new_labels = []
        animal_images = []
        machine_images = []
        for label in labels:
            if label in animal_labels:
                new_labels.append(1)
            elif label in machine_labels:
                new_labels.append(0)
            else:
                print("Error in assigning labels")
        return np.array(new_labels)

    if binary_classification:
        test_labels = convert_labels(test_labels)
        train_labels = convert_labels(train_labels)

    num_train = math.floor(len(train_labels)*subset_size)
    num_test = math.floor(len(test_labels)*subset_size)

    test_labels = test_labels[:num_test]
    train_labels = train_labels[:num_train]
    test_images = test_images[:num_test]
    train_images = train_images[:num_train]

    num_batches = math.ceil(num_train/batch_size)
    batch_size = int(num_train/num_batches)

    print("Adjusted batch size to " + str(batch_size))

    def plotImages(images_arr):
        fig, axes = plt.subplots(1, 5, figsize=(20,20))
        axes = axes.flatten()
        for img, ax in zip( images_arr, axes):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    #plotImages(train_images[:5])

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
        Dense(1)
    ])

    if binary_classification:
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])
    else:
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossEntropy(from_logits=True),
                      metrics=['accuracy'])

    model.summary()

    history = model.fit(
        train_images, train_labels,
        steps_per_epoch=num_train // batch_size,
        epochs=epochs,
        validation_data=(test_images, test_labels),
        validation_steps=num_test // batch_size
    )

    ## Augment dataset with unlabeled images

    if self_train:
        unlabeled_images = read_all_images(binary_path + "unlabeled_X.bin")
        unlabeled_predictions = model.predict(unlabeled_images)
        for i in range(len(unlabeled_predictions)):
            if i % 50 == 0:
                print("Augmenting dataset " + str(i) + "/" + str(len(unlabeled_images)) + " complete")
            pred = unlabeled_predictions[i]
            image = unlabeled_images[i]
            if np.amax(pred) >= .75:
                train_images = np.append(train_images, image)
                train_labels = np.append(train_labels, np.argmax(pred))
                # train_images.append(image)
                # converted_train_labels.append(np.argmax(pred))

        history = model.fit(
            train_images, train_labels,
            steps_per_epoch=num_train // batch_size,
            epochs=epochs,
            validation_data=(test_images, test_labels),
            validation_steps=num_test // batch_size
        )

    model.save(name)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    print(acc)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

if __name__ == "__main__":
   main(sys.argv[1:])
