import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data.stl10_input import read_labels, read_all_images

import matplotlib.pyplot as plt
import numpy as np

batch_size = 100
epochs = 15
IMG_HEIGHT = 96
IMG_WIDTH = 96

binary_path = "data/stl10_binary/"

train_images = read_all_images(binary_path + "train_x.bin")
train_labels = read_labels(binary_path + "train_Y.bin")

test_images = read_all_images(binary_path + "test_x.bin")
test_labels = read_labels(binary_path + "test_Y.bin")
unlabeled_images = read_all_images(binary_path + "unlabeled_X.bin")[1:10000]


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

converted_test_labels = convert_labels(test_labels)
converted_train_labels = convert_labels(train_labels)

num_train = len(train_labels)
num_test = len(test_labels)

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

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

model.fit(
    train_images, converted_train_labels,
    steps_per_epoch=num_train // batch_size,
    epochs=1,
    validation_data=(test_images, converted_test_labels),
    validation_steps=num_test // batch_size
)

## Augment dataset with unlabeled images
threshold = .75
unlabeled_predictions = model.predict(unlabeled_images)
for i in range(len(unlabeled_predictions)):
    pred = unlabeled_predictions[i]
    image = unlabeled_images[i]
    if np.amax(pred) >= threhold:
        train_images = np.append(train_images,image)
        converted_train_labels = np.append(converted_train_labels,np.argmax(pred))

history = model.fit(
    train_images, converted_train_labels,
    steps_per_epoch=num_train // batch_size,
    epochs=1,
    validation_data=(test_images, converted_test_labels),
    validation_steps=num_test // batch_size
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
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

