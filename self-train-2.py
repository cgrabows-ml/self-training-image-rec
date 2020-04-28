import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras import regularizers


from data.stl10_input import read_labels, read_all_images, read_some_images

import matplotlib.pyplot as plt
import numpy as np
import math
import os

import sys, getopt

from util import convert_labels



def main(argv):

    opts, args = getopt.getopt(argv, "hi:o:", ["test=", "experiment=", "subset=", "self_train=", "epochs=", "batch=", "threshold="])

    # Setting Defaults:
    self_train = False
    subset_size = 1
    epochs = 15
    batch_size = 1000
    threshold = .75
    experiment = "default"
    binary_classification = False


    # Reading Options
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

    print("Self Train: ", self_train)
    print("Subset Size: ", subset_size)
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

    print("Test Labels: ", len(test_labels))
    print("Train Labels: ", len(train_labels))

    # Converting labels - we split to 3 label sets with disagreement
    if experiment == "disagreement":
        # FILL ME IN
        # 3 label sets? or just 1 and then additional code later
    else:
        test_labels, test_images = convert_labels(test_labels, test_images, experiment)
        train_labels, train_images = convert_labels(train_labels, train_images, experiment)
    print("Test Label Shape: ", test_labels.shape)
    print("Test Label Shape: ", train_labels.shape)
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


    # Defining Model: Fill in improved model here

    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu',
               input_shape=(IMG_HEIGHT, IMG_WIDTH ,3),
               kernel_regularizer=regularizers.l2(0.01)),
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

    if experiment == "disagreement":
        Amodel = Sequential([
            Conv2D(16, 3, padding='same', activation='relu',
                   input_shape=(IMG_HEIGHT, IMG_WIDTH ,3),
                   kernel_regularizer=regularizers.l2(0.01)),
            MaxPooling2D(),
            Dropout(0.2),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Dropout(0.2),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(7)
        ])
        Mmodel = Sequential([
            Conv2D(16, 3, padding='same', activation='relu',
                   input_shape=(IMG_HEIGHT, IMG_WIDTH ,3),
                   kernel_regularizer=regularizers.l2(0.01)),
            MaxPooling2D(),
            Dropout(0.2),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Dropout(0.2),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(5)
        ])

    if experiment=="binary":
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])
    elif experiment == "disagreement":
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        Amodel.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        Mmodel.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
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

    if experiment == "disagreement":
        # Single loop.
        if not self_train:
            # history = model.fit(
            #     aug.flow(train_images, train_labels, batch_size = batch_size),
            #     steps_per_epoch=num_train // batch_size,
            #     epochs=epochs,
            #     callbacks=[cp_callback],
            #     validation_data=(test_images, test_labels),
            #     validation_steps=num_test // batch_size
            # )
            Ahistory = Amodel.fit(
                aug.flow(train_images, train_labels, batch_size = batch_size),
                steps_per_epoch=num_train // batch_size,
                epochs=epochs,
                callbacks=[cp_callback],
                validation_data=(test_images, test_labels),
                validation_steps=num_test // batch_size
            )
            Mhistory = Mmodel.fit(
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
                                                int(100000 * subset_size))
            confident = True
            # While there are still confident labelings
            while confident:
                # First train supervised model
                print("Train images and Label shape: ",
                      train_images.shape, train_labels.shape)
                # history = model.fit(
                #     aug.flow(train_images, train_labels, batch_size = batch_size),
                #     steps_per_epoch=num_train // batch_size,
                #     epochs=epochs,
                #     callbacks=[cp_callback],
                #     validation_data=(test_images, test_labels),
                #     validation_steps=num_test // batch_size
                # )
                Ahistory = Amodel.fit(
                    aug.flow(train_images, train_labels, batch_size = batch_size),
                    steps_per_epoch=num_train // batch_size,
                    epochs=epochs,
                    callbacks=[cp_callback],
                    validation_data=(test_images, test_labels),
                    validation_steps=num_test // batch_size
                )
                Mhistory = Mmodel.fit(
                    aug.flow(train_images, train_labels, batch_size = batch_size),
                    steps_per_epoch=num_train // batch_size,
                    epochs=epochs,
                    callbacks=[cp_callback],
                    validation_data=(test_images, test_labels),
                    validation_steps=num_test // batch_size
                )
                # print(train_images[0])


                # Predict unlabeled examples and add confident ones
                count = 0
                if (len(unlabeled_images) == 0): break
                unlabeled_predictionsA = Amodel.predict(unlabeled_images)
                unlabeled_predictionsM = Mmodel.predict(unlabeled_images)
                # Converting this to probabilities:
                # print(unlabeled_predictions, unlabeled_predictions.shape)
                probs_A = tf.nn.softmax(unlabeled_predictionsA)
                probs_M = tf.nn.softmax(unlabeled_predictionsM)
                temp_labA = np.argmax(probs_A, axis = 1)
                temp_labM = np.argmax(probs_M, axis = 1)

                # Combining animal and machine labels to one
                animal_labels = [2,4,5,6,7,8,0]
                machine_labels = [1,3,9,10,0]
                # If both predicts other or both predict non-other,
                # it should not get selected for ST so wrong labels wont matter
                new_labels = []
                for i in range(len(temp_labA)):
                    new_labels.append(max(animal_labels[temp_labA[i]],
                                          machine_labels[temp_labM[i]]))
                class_predsA = np.amax(probs_A, axis = 1)
                class_predsM = np.amax(probs_M, axis = 1)
                class_predsA = tf.reshape(class_predsA, [len(class_predsA)])
                class_predsM = tf.reshape(class_predsM, [len(class_predsM)])
                class_preds = class_predsA * class_predsM

                assert class_preds.shape == class_predsA.shape

                to_remove = (class_preds >= threshold) and
                    ((temp_labA == 6) != (temp_labM == 4))
                    # XOR: one and only one of the predictions is 'other'
                train_images = np.append(train_images, unlabeled_images[to_remove])
                train_labels = np.append(train_labels, new_labels[to_remove])
                count = np.sum(to_remove)
                unlabeled_images = unlabeled_images[np.logical_not(to_remove)]
                print(count)
                print(unlabeled_images.shape)

                #train_images = train_images.reshape(-1, 3, 96, 96)
                #train_images = np.transpose(train_images, (0, 3, 2, 1))
                train_images = train_images.reshape(-1, 96, 96, 3)
                #print(train_images[0])

                # Recalculating num_train and batch_size:
                num_train = len(train_labels)
                num_batches = math.ceil(num_train/batch_size)
                batch_size = int(num_train/num_batches)

                # No confident labelings left
                if count == 0: #< 50:
                    confident = False
                    print("Exiting loop")
    else:
        # Single loop.
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
                                                int(100000 * subset_size))
            confident = True
            # While there are still confident labelings
            while confident:
                # First train supervised model
                print("Train images and Label shape: ",
                      train_images.shape, train_labels.shape)
                history = model.fit(
                    aug.flow(train_images, train_labels, batch_size = batch_size),
                    steps_per_epoch=num_train // batch_size,
                    epochs=epochs,
                    callbacks=[cp_callback],
                    validation_data=(test_images, test_labels),
                    validation_steps=num_test // batch_size
                )
               # print(train_images[0])
                # Predict unlabeled examples and add confident ones
                count = 0
                if (len(unlabeled_images) == 0): break
                unlabeled_predictions = model.predict(unlabeled_images)
                # Converting this to probabilities:
                temp_lab = -1
                if experiment != "binary":
                    probss = tf.nn.softmax(unlabeled_predictions)
                    class_preds = np.amax(probss, axis = 1)
                    temp_lab = np.argmax(probss, axis = 1)
                else:
                    # print(unlabeled_predictions, unlabeled_predictions.shape)
                    class_preds = tf.nn.sigmoid(unlabeled_predictions)
                    temp_lab = tf.round(class_preds)
                class_preds = tf.reshape(class_preds, [len(class_preds)])

                # Version 1: Not vectorized
                # to_keep = list(range(len(class_preds)))
                # for i in range(len(class_preds)):
                #     if i % 50 == 0:
                #         print("Augmenting dataset " + str(i) + "/" + str(len(unlabeled_images)) + " complete")
                #     pred = class_preds[i]
                #     image = unlabeled_images[i]
                #     if np.amax(pred) >= threshold:
                #         train_images = np.append(train_images, image)
                #         train_labels = np.append(train_labels, temp_lab[i])
                #         # decrease size of unlabeled images
                #         to_keep.remove(i)
                #         count += 1
                #unlabeled_images = unlabeled_images[to_keep, :, :, :]

                # Version 2: Vectorized?
                to_remove = class_preds >= threshold
                train_images = np.append(train_images, unlabeled_images[to_remove])
                train_labels = np.append(train_labels, temp_lab[to_remove])
                count = np.sum(to_remove)
                unlabeled_images = unlabeled_images[np.logical_not(to_remove)]
                print(count)
                print(unlabeled_images.shape)

                #train_images = train_images.reshape(-1, 3, 96, 96)
                #train_images = np.transpose(train_images, (0, 3, 2, 1))
                train_images = train_images.reshape(-1, 96, 96, 3)
                #print(train_images[0])
                # Recalculating num_train and batch_size:
                num_train = len(train_labels)
                num_batches = math.ceil(num_train/batch_size)
                batch_size = int(num_train/num_batches)

                # No confident labelings left
                if count == 0: #< 50:
                    confident = False
                    print("Exiting loop")
    model.save_weights(checkpoint_path.format(epoch=0))
    # model.save(expe)




    # TO DO: computing accuracy of model and loss of model
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
