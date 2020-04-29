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
    IMG_HEIGHT = 96
    IMG_WIDTH = 96

    binary_path = "data/stl10_binary/"

    # if experiment != "disagreement":

    og_train_images = read_all_images(binary_path + "train_X.bin")
    og_train_labels = read_labels(binary_path + "train_y.bin")

    og_test_images = read_all_images(binary_path + "test_X.bin")
    og_test_labels = read_labels(binary_path + "test_y.bin")

    experiments = []
    if experiment == "disagreement":
        experiments = ["disagreement_animal", "disagreement_machine"]
    else:
        experiments = [experiment]

    models = []
    all_train_labels = []
    all_test_labels = []

    for sub_experiment in experiments:

        checkpoint_path = "training/" + experiment + "-cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create a callback that saves the model's weights every 5 epochs
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            period=5)

        test_labels, test_images = convert_labels(og_test_labels, og_test_images, sub_experiment)
        train_labels, train_images = convert_labels(og_train_labels, og_train_images, sub_experiment)

        num_train = math.floor(len(train_labels)*subset_size)
        num_test = math.floor(len(test_labels)*subset_size)

        test_labels = test_labels[:num_test]
        train_labels = train_labels[:num_train]
        test_images = test_images[:num_test]
        train_images = train_images[:num_train]
        all_train_labels.append(train_labels)
        all_test_labels.append(test_labels)

        print(num_train, batch_size)

        num_batches = math.ceil(num_train/batch_size)
        batch_size = int(num_train/num_batches)

        print("Adjusted batch size to " + str(batch_size))

        if sub_experiment == "binary":
            num_classes = 1
        elif sub_experiment == "animal":
            num_classes = 6
        elif sub_experiment == "machine":
            num_classes = 4
        elif sub_experiment == "disagreement_animal":
            num_classes = 7
        elif sub_experiment == "disagreement_machine":
            num_classes = 5
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

        history = model.fit(
            # fixed
            train_images, train_labels,
            # broken
            # aug.flow(train_images, train_labels, batch_size = batch_size),
            steps_per_epoch=num_train // batch_size,
            epochs=epochs,
            callbacks=[cp_callback],
            validation_data=(test_images, test_labels),
            validation_steps=num_test // batch_size
        )

        if experiment == "disagreement":
            model.save_weights((checkpoint_path + sub_experiment).format(epoch=0))
            models.append(model)
        else:
            model.save_weights(checkpoint_path.format(epoch=0))
            models.append(model)

    aug = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")

    if self_train:

        if experiment == "disagreement":
            ## Augment dataset with unlabeled images

            animal_model = models[0]
            machine_model = models[1]
            animal_train_labels = all_train_labels[0]
            machine_train_labels = all_train_labels[1]
            animal_test_labels = all_test_labels[0]
            machine_test_labels = all_test_labels[1]

            # Reading only 1000 images for lack of computing power
            #unlabeled_images = read_all_images(binary_path + "unlabeled_X.bin")
            unlabeled_images = read_some_images(binary_path + "unlabeled_X.bin",
                                                int(100000 * subset_size))
            confident = True
            # While there are still confident labelings
            while confident:
                # First train supervised model
                print("Train images and Animal, machine Label shape: ",
                      train_images.shape, animal_train_labels.shape, machine_train_labels.shape)
                print("test images and Animal, machine Label shape: ",
                      test_images.shape, animal_test_labels.shape, machine_test_labels.shape)
                # history = model.fit(
                #     aug.flow(train_images, train_labels, batch_size = batch_size),
                #     steps_per_epoch=num_train // batch_size,
                #     epochs=epochs,
                #     callbacks=[cp_callback],
                #     validation_data=(test_images, test_labels),
                #     validation_steps=num_test // batch_size
                # )
                animal_history = animal_model.fit(
                    aug.flow(train_images, animal_train_labels, batch_size = batch_size),
                    steps_per_epoch=num_train // batch_size,
                    epochs=epochs,
                    callbacks=[cp_callback],
                    validation_data=(test_images, animal_test_labels),
                    validation_steps=num_test // batch_size
                )
                machine_history = machine_model.fit(
                    aug.flow(train_images, machine_train_labels, batch_size = batch_size),
                    steps_per_epoch=num_train // batch_size,
                    epochs=epochs,
                    callbacks=[cp_callback],
                    validation_data=(test_images, machine_test_labels),
                    validation_steps=num_test // batch_size
                )
                # print(train_images[0])


                # Predict unlabeled examples and add confident ones
                count = 0
                if (len(unlabeled_images) == 0): break
                unlabeled_predictionsA = animal_model.predict(unlabeled_images)
                unlabeled_predictionsM = machine_model.predict(unlabeled_images)
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
                new_animal_labels = []
                new_machine_labels = []
                for i in range(len(temp_labA)):
                    new_animal_labels.append(temp_labA[i])
                    new_machine_labels.append(temp_labM[i])
                    # new_labels.append(max(animal_labels[temp_labA[i]],
                    #                       machine_labels[temp_labM[i]]))
                class_predsA = np.amax(probs_A, axis = 1)
                class_predsM = np.amax(probs_M, axis = 1)
                class_predsA = tf.reshape(class_predsA, [len(class_predsA)])
                class_predsM = tf.reshape(class_predsM, [len(class_predsM)])
                class_preds = class_predsA * class_predsM

                assert class_preds.shape == class_predsA.shape
                print("pred shape", class_predsA.shape)

                to_remove_thresh = class_preds >= threshold
                to_remove_xor = (temp_labA == 6) != (temp_labM == 4)
                    # XOR: one and only one of the predictions is 'other'
                to_remove = np.logical_and(to_remove_thresh, to_remove_xor)
                train_images = np.append(train_images, unlabeled_images[to_remove])
                # print(new_a / nimal_labels.dtype)
                new_animal_labels = np.array(new_animal_labels)[to_remove]
                new_machine_labels = np.array(new_machine_labels)[to_remove]
                animal_train_labels = np.append(animal_train_labels, new_animal_labels)
                machine_train_labels = np.append(machine_train_labels, new_machine_labels)
                count = np.sum(to_remove)
                unlabeled_images = unlabeled_images[np.logical_not(to_remove)]
                print(count)
                print(unlabeled_images.shape)

                #train_images = train_images.reshape(-1, 3, 96, 96)
                #train_images = np.transpose(train_images, (0, 3, 2, 1))
                train_images = train_images.reshape(-1, 96, 96, 3)

                #print(train_images[0])

                # Recalculating num_train and batch_size:
                num_train = len(animal_train_labels)
                num_batches = math.ceil(num_train/batch_size)
                batch_size = int(num_train/num_batches)

                # No confident labelings left
                if count == 0: #< 50:
                    confident = False
                    print("Exiting loop")
        else:
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
