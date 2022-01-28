
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from tensorflow.keras.layers import BatchNormalization

# Initialising the CNN
classifier = Sequential()

# 1st Convolutional Layer

classifier.add(Conv2D(8, kernel_size=(5, 5), padding='same',activation='relu',input_shape=(224,224,3),strides=(4,4)))
classifier.add(BatchNormalization())

# 2nd Convolutional Layer
classifier.add(Conv2D(8, kernel_size=(3, 3),activation='relu'))
classifier.add(BatchNormalization())

# 3rd Convolutional Layer
classifier.add(Conv2D(8, kernel_size=(3, 3),activation='relu'))
classifier.add(BatchNormalization())

# 4th Convolutional Layer
classifier.add(Conv2D(96, kernel_size=(1, 1), padding='same',activation='relu'))
classifier.add(MaxPooling2D(pool_size=(3, 3)))
classifier.add(BatchNormalization())

# 5th Convolutional Layer
classifier.add(Conv2D(8, kernel_size=(3, 3),activation='relu',padding='same'))
classifier.add(BatchNormalization())

# 6th Convolutional Layer
classifier.add(Conv2D(8, kernel_size=(3, 3),activation='relu',padding='same'))
classifier.add(BatchNormalization())

# 7th Convolutional Layer
classifier.add(Conv2D(8, kernel_size=(3, 3),activation='relu',padding='same'))
classifier.add(BatchNormalization())

# 8th Convolutional Layer
classifier.add(Conv2D(8, kernel_size=(3, 3),activation='relu',padding='same'))
classifier.add(BatchNormalization())

# 9th Convolutional Layer
classifier.add(Conv2D(256, kernel_size=(1, 1),activation='relu',padding='same'))
classifier.add(MaxPooling2D(pool_size=(3, 3)))
classifier.add(BatchNormalization())

# 10th Convolutional Layer
classifier.add(Conv2D(8, kernel_size=(3, 3),activation='relu',padding='same'))
classifier.add(BatchNormalization())

# 11th Convolutional Layer
classifier.add(Conv2D(384, kernel_size=(1, 1),activation='relu',padding='same'))
classifier.add(BatchNormalization())

# 12th Convolutional Layer
classifier.add(Conv2D(8, kernel_size=(3, 3),activation='relu',padding='same'))
classifier.add(BatchNormalization())

# 13th Convolutional Layer
classifier.add(Conv2D(384, kernel_size=(1, 1),activation='relu',padding='same'))
classifier.add(BatchNormalization())

# 14th Convolutional Layer
classifier.add(Conv2D(8, kernel_size=(3, 3),activation='relu',padding='same'))
classifier.add(BatchNormalization())

# 15th Convolutional Layer
classifier.add(Conv2D(256, kernel_size=(1, 1),activation='relu',padding='same'))
classifier.add(MaxPooling2D(pool_size=(3, 3)))
classifier.add(BatchNormalization())


# Passing it to a dense layer
classifier.add(Flatten())


# 1st Dense Layer
classifier.add(Dense(20,activation='relu'))

classifier.add(Dropout(0.4))
classifier.add(BatchNormalization())


# Output Layer
classifier.add(Dense(5,activation='softmax'))


classifier.summary()

# (4) Compile 
classifier.compile(loss='CategoricalCrossentropy', optimizer='adam',metrics=['accuracy'])
 
# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   brightness_range=(0.5,1),
                                   horizontal_flip = True,
                                   vertical_flip=True,
                                   validation_split=0.3)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('flowers',
                                                 target_size = (224, 224),
                                                 batch_size = 16,
                                                 shuffle=True,
                                                 subset="training")

test_set = test_datagen.flow_from_directory('flowers',
                                            target_size = (224, 224),
                                            batch_size = 16,
                                            shuffle=True,
                                            subset="validation")

model = classifier.fit_generator(training_set,
                         steps_per_epoch = 8,
                         epochs = 10,
                         validation_data = test_set,    
                         validation_steps = 5)

classifier.save("model.h5")
print("Saved model to disk")





