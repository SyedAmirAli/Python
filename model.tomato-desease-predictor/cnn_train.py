
print("\nTraining Started...\n")

# import Keras packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np
from keras.utils import plot_model  # Change the import statement

# Initializing the CNN
np.random.seed(1337)
classifier = Sequential()

classifier.add(Convolution2D(32, 3, 3, input_shape=(128, 128, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Convolution2D(16, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Convolution2D(8, 3, 3, activation='relu'))

# Add a condition to check spatial dimensions before applying MaxPooling
if classifier.output_shape[1] > 1 and classifier.output_shape[2] > 1:
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

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

#Part 2 - fitting the data set

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'train',
        target_size=(128, 128),
        batch_size=64,
        class_mode='categorical' )
label_map = (training_set.class_indices)

print(label_map)

test_set = test_datagen.flow_from_directory(
        'val',
        target_size=(128, 128),
        batch_size=64,
        class_mode='categorical')


classifier.fit_generator(
        training_set,
        steps_per_epoch=70,
        epochs=500,
        validation_data=test_set,
        validation_steps=50
)

# classifier.save_weights('model/keras_potato_trained_model_weights.h5')
# Save the model in native Keras format
classifier.save('model-v2/tomato_disease.keras')
classifier.save('model-v2/tomato_disease.h5')
classifier.save('model-v2/tomato_disease', save_format='tf')
print('\nSaved trained model as %s ' % 'tomato_disease.h5\n')
