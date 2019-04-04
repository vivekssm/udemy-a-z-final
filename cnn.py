# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential  #same as ann.. to initialize neural network
from keras.layers import Convolution2D #used to make first step i.e convolution step to our NN. images in 2D so 2D
from keras.layers import MaxPooling2D  #to add pooling layer
from keras.layers import Flatten   # to add flattening layer output of this layer act as input to futher layers of NN
from keras.layers import Dense  # same as ANN .. TO ADD SAVERAL LAYERS TO NETWORK

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
#no. of feature maps we want to create=32
#no of rows in feature detector=3
#no of column in feature detector=3
#shape of input image=64*64*3 (coloured image hai islie 3-D) jo image nahi v hoga wo convert ho jaega is shape me
#GPU pe kam karenge to feature map 64*3*3 le sakte aur image 128*128*3 le sakte
#activation function=RELU
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
#(yaha 2*2 matrix le k max pooling karenge)
# jyada bada matrix lene pe information lose kar sakte hai.
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer     (to improve accuracy) 
                         #input aega pichhle step se.. we don't need to explain

classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection  #ann ki tarah
      #hidden layer me 128 nodes & RELU activation function
      #output me 1 node, activation is sigmoid
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN (explanation from ann)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#cnn is already built
# Part 2 - Fitting the CNN to the images (image augmentation) (read copy) 
#preprocess the image to avoid overfitting

# below is the code for image augmentation

from keras.preprocessing.image import ImageDataGenerator
                    # imaage me kya kya change karna. i.e, rescale,shear_range,zoom_range,horizontal_flip
                    
                    #yaha hum train_datagen define kie
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

                    #yaha hum test_datagen define kie

test_datagen = ImageDataGenerator(rescale = 1./255)
                    
                    #yaha hum train_datagon ko apply kie training_set pe
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),   #image size expected from CNN
                                                 batch_size = 32,       #kitne image ka batch banana hai
                                                 class_mode = 'binary')


                    #yaha hum test_datagon ko define kie test-set pe
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

#hamare CNN  ka nam hai classifier ham uspe ye sab fit kar rahe hai..
classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,     #here all images i.e 8000 images will pass through CNN in each epoch
                         nb_epoch = 25,      #no. of epochs we want
                         validation_data = test_set,
                         nb_val_samples = 2000)  # no of images in test set