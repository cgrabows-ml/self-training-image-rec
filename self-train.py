import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


from data.stl10_input import read_labels, read_all_images

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

        # aug = ImageDataGenerator(
        #     rotation_range=30,
        #     zoom_range=0.15,
        #     width_shift_range=0.2,
        #     height_shift_range=0.2,
        #     shear_range=0.15,
        #     horizontal_flip=True,
        #     fill_mode="nearest")

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


    ## Augment dataset with unlabeled images
    animal_labels = [2,4,5,6,7,8]
    machine_labels = [1,3,9,10]
    animal_model = models[0]
    machine_model = models[1]

    if self_train:
        unlabeled_images = read_all_images(binary_path + "unlabeled_X.bin")
        if experiment == "disagreement":
            animal_predictions = animal_model.predict(unlabeled_images)
            machine_predictions = machine_model.predict(unlabeled_images)
            for i in range(len(unlabeled_images)):
                best_pred = 0
                pred = -1
                for j in range(10):
                    label = j+1
                    if label in animal_labels:
                        possible_pred = animal_predictions[i][animal_labels.index(label)]*machine_predictions[len(machine_labels)]
                    elif label in machine_labels:
                        possible_pred = machine_predictions[i][machine_labels.index(label)]*animal_predictions[len(animal_labels)]
                    if possible_pred > best_pred:
                        best_pred = possible_pred
                        pred = label

                if i % 50 == 0:
                    print("Augmenting dataset " + str(i) + "/" + str(len(unlabeled_images)) + " complete")
                # pred = unlabeled_predictions[i]
                image = unlabeled_images[i]
                if best_pred >= .75:
                    train_images = np.append(train_images, image)
                    # train_labels = np.append(train_labels, np.argmax(pred))
                    animal_train_labels = np.append(animal_train_labels, pred)
                    machine_train_labels = np.append(machine_train_labels, pred)
                    # train_images.append(image)
                    # converted_train_labels.append(np.argmax(pred))

            animal_history = animal_model.fit(
                # fixed
                train_images, animal_train_labels,
                # broken
                # aug.flow(train_images, train_labels, batch_size = batch_size),
                steps_per_epoch=num_train // batch_size,
                epochs=epochs,
                validation_data=(test_images, test_labels),
                validation_steps=num_test // batch_size
            )
            animal_model.save_weights((checkpoint_path + "disagreement_animal").format(epoch=0))

            machine_history = animal_model.fit(
                # fixed
                train_images, animal_train_labels,
                # broken
                # aug.flow(train_images, train_labels, batch_size = batch_size),
                steps_per_epoch=num_train // batch_size,
                epochs=epochs,
                validation_data=(test_images, test_labels),
                validation_steps=num_test // batch_size
            )
            machine_model.save_weights((checkpoint_path + "disagreement_machine").format(epoch=0))


        else:
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
                # fixed
                train_images,train_labels,
                # broken
                # aug.flow(train_images, train_labels, batch_size = batch_size),
                steps_per_epoch=num_train // batch_size,
                epochs=epochs,
                validation_data=(test_images, test_labels),
                validation_steps=num_test // batch_size
            )
            model.save_weights(checkpoint_path.format(epoch=0))



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
