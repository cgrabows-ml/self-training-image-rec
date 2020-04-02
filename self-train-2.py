import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


from data.stl10_input import read_labels, read_all_images, read_some_images

import matplotlib.pyplot as plt
import numpy as np
import math
import os

import sys, getopt

from util import convert_labels



def main(argv):

    opts, args = getopt.getopt(argv, "hi:o:", ["test=", "experiment=", "subset=", "self_train=", "epochs=", "batch=", "threshold="])

    self_train = False
    subset_size = 1
    epochs = 15
    batch_size = 1000
    threshold = .75
    experiment = "default"
    binary_classification = False

    for opt, arg in opts:
        if opt == '-h':
            print('self-train.py --experiment=binary --subset=1 --self_train=True --epochs=20 --batch=1000 --threshold=.75')
            sys.exit()
        elif opt in ("-st", "--self_train"):
            print(arg)
            self_train= arg.lower() in ["true", "True"]
        elif opt in ("-ss", "--subset"):
            subset_size = float(arg)
        elif opt in ("-exp", "--experiment"):
            experiment = arg
            if experiment=="binary":
                binary_classification = True
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-b", "--batch"):
            batch_size = int(arg)
        elif opt in ("-t", "--threshold"):
            threshold = float(arg)

    print(self_train)
    print(subset_size)
    IMG_HEIGHT = 96
    IMG_WIDTH = 96

    binary_path = "data/stl10_binary/"

    train_images = read_all_images(binary_path + "train_X.bin")
    train_labels = read_labels(binary_path + "train_y.bin")

    test_images = read_all_images(binary_path + "test_X.bin")
    test_labels = read_labels(binary_path + "test_y.bin")


    checkpoint_path = "training/" + experiment + "-cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        period=5)

    print(len(test_labels))
    print(len(train_labels))
    test_labels, test_images = convert_labels(test_labels, test_images, experiment)
    train_labels, train_images = convert_labels(train_labels, train_images, experiment)
    print(test_labels.shape)
    print(train_labels.shape)
    # exit()


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

    if experiment == "binary":
        num_classes = 1
    elif experiment == "animal":
        num_classes = 6
    elif experiment == "machine":
        num_classes = 4
    else:
        num_classes = 10

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

    if experiment=="binary":
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])
    else:
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

    model.summary()

    aug = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")

    if not self_train:
        history = model.fit(
            aug.flow(train_images, train_labels, batch_size = batch_size),
            steps_per_epoch=num_train // batch_size,
            epochs=epochs,
            callbacks=[cp_callback],
            validation_data=(test_images, test_labels),
            validation_steps=num_test // batch_size
        )



    ## Augment dataset with unlabeled images

    if self_train:
        # Reading only 1000 images for lack of computing power
        #unlabeled_images = read_all_images(binary_path + "unlabeled_X.bin")
        unlabeled_images = read_some_images(binary_path + "unlabeled_X.bin",
                                            1000)
        confident = True
        # While there are still confident labelings
        while confident:
            # First train supervised model
            history = model.fit(
                aug.flow(train_images, train_labels, batch_size = batch_size),
                steps_per_epoch=num_train // batch_size,
                epochs=epochs,
                validation_data=(test_images, test_labels),
                validation_steps=num_test // batch_size
            )

            # Predict unlabeled examples and add confident ones
            count = 0
            unlabeled_predictions = model.predict(unlabeled_images)
            to_keep = list(range(len(unlabeled_predictions)))
            for i in range(len(unlabeled_predictions)):
                if i % 50 == 0:
                    print("Augmenting dataset " + str(i) + "/" + str(len(unlabeled_images)) + " complete")
                pred = unlabeled_predictions[i]
                image = unlabeled_images[i]
                if np.amax(pred) >= threshold:
                    train_images = np.append(train_images, image)
                    train_labels = np.append(train_labels, pred.argmax(axis=-1))
                    # decrease size of unlabeled images
                    to_keep.remove(i)
                    count = 1
            unlabeled_images = unlabeled_images[to_keep, :, :, :]
            train_images = train_images.reshape(-1, 3, 96, 96)
            # Recalculating num_train and batch_size:
            num_train = len(train_labels)
            num_batches = math.ceil(num_train/batch_size)
            batch_size = int(num_train/num_batches)

            # No confident labelings left
            if count == 0:
                confident = False
    model.save_weights(checkpoint_path.format(epoch=0))
    # model.save(expe)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    print("Traing Accuracy:", acc)
    print("Val Accuracy", val_acc)
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
