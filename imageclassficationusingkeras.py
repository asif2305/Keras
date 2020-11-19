from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras import backend as K
import numpy as np
from keras.preprocessing import image

#dimensions of our images.
img_width,img_height=150,150

train_data_dir='data/train' # dataset directory
validation_data_dir='data/validation' # val directory
nb_train_samples=100 # no of images #
nb_validation_samples=100 # no of validation image
epochs=2 # looping
batch_size=20 # 50= train_samples/batch_size -- how many pic we need at the same point of time
if K.image_data_format()=='channels_first':
    input_shape=(3,img_width,img_height) # 150,150,3
else:
    input_shape=(img_width,img_height,3)

#augmentation
train_datagen=ImageDataGenerator(
    rescale=1. /255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
# this is the augmentation configuration we will use for testing
# only rescaling
test_datagen=ImageDataGenerator(rescale=1. /255)

train_generator=train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width,img_height),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator=test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width,img_height),
    batch_size=batch_size,
    class_mode='binary'
)
# now make neural network model #################################################################################
# first time
model=Sequential() # create a object for sequential
# convolution network means when we extract features fom image
model.add(Conv2D(32,(3,3),input_shape=input_shape)) # 32 is a features,(3,3) is a matrix,input shape is a image size
model.add(Activation('relu'))
# pooling means it is reduce the size of image without reducing the features of the image
# we have so many features  which we do not need.thats why we use MaxPooling2D to get the required features which we need.
model.add(MaxPooling2D(pool_size=(2,2)))
# 2nd time
model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# third time
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# now we will use flatten which make the 2D image into 1D image
model.add(Flatten())
# Dense means how many input or output we need
model.add(Dense(64)) # input node
model.add(Activation('relu')) # in relu data will come either 1 r 0
model.add(Dropout(0.5))
model.add(Dense(1)) # output node
model.add(Activation('sigmoid')) # to get the specific data

model.summary()

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

# this is the augmentation configuration we will use for training

#execution of neural network model

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples, # batch size
    epochs=epochs, # looping
    validation_data=validation_generator,
    validation_steps=nb_validation_samples  # batch size
    )
model.save_weights('first_try.h5') # save all node and weight in neural network

# convert image to numpy array for prediction
img_pred=image.load_img('data/validation/dogs/cat.496.jpg',target_size=(150,150))
img_pred=image.img_to_array(img_pred)
img_pred=np.expand_dims(img_pred,axis=0)


########################################################################### for prediction

rstl=model.predict(img_pred)
print(rstl)
if rstl[0][0]==1:
    prediction='dog'
else:
    prediction='cat'

print(prediction)













