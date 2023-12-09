# Part 1: Building a CNN

# import Keras packages
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
from keras.utils import plot_model

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

# Initializing the CNN
np.random.seed(1337)
classifier = Sequential()

classifier.add(Convolution2D(32, 3, 3, input_shape=(128, 128, 3), activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
classifier.add(Convolution2D(16, 3, 3, activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
classifier.add(Convolution2D(8, 3, 3, activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

classifier.add(Flatten())

# hidden layer
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(rate=0.5))

# output layer
classifier.add(Dense(units=10, activation='softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(classifier.summary())

# Plot the model
plot_model(classifier, show_shapes=True, to_file='PlantVillage_CNN.png')

# Part 2 - fitting the data set
from keras.preprocessing.image import ImageDataGenerator

# Assuming project_directory is defined earlier in your code
project_directory = 'E:/Amir/Python/model.tomato-diseases-dectector'

train_directory = os.path.join(project_directory, 'train')
val_directory = os.path.join(project_directory, 'val')

print(os.listdir(train_directory))
print(os.listdir(val_directory))


# Check if the directories exist
for directory in [train_directory, val_directory]:
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Update paths for flow_from_directory
training_set = train_datagen.flow_from_directory(
    train_directory,
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical'
)
label_map = training_set.class_indices
print(label_map)

test_set = test_datagen.flow_from_directory(
    val_directory,
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical'
)

# Check if there are any images in the directories
if len(training_set) == 0 or len(test_set) == 0:
    raise ValueError("No images found in the specified directories.")

# Fix fit_generator call
classifier.fit_generator(
    training_set,
    steps_per_epoch=len(training_set),
    epochs=1000,
    validation_data=test_set,
)

# classifier.save_weights('keras_potato_trained_model_weights.h5')
classifier.save(os.path.join(project_directory, 'tomato-model', 'model.h5'))
classifier.save(os.path.join(project_directory, 'tomato-model'), save_format='tf')
print('\nSaved trained model as %s ' % 'model.h5\n')
