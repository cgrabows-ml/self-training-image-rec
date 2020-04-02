import numpy as np

def convert_labels(labels, images, experiment):
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
            elif labels[i] in machine_labels:
                if experiment == "machine":
                    new_labels.append(machine_labels.index(labels[i]))
                    new_images.append(images[i])
                elif experiment == "binary":
                    new_labels.append(0)
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