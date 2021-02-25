######### Import Libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential                # sequential model to add layers behind some
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten, Dropout, MaxPool2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam


####### Load Data
# img_width, img_height = 150, 150
# train_datagen = ImageDataGenerator(
#                  rescale=1./255,
#                  rotation_range=30,
#                  shear_range=0.3,
#                  zoom_range=0.3,
#                  width_shift_range=0.4,
#                  height_shift_range=0.4,
#                  horizontal_flip=True,
#                  fill_mode='nearest')
#
# test_datagen = ImageDataGenerator(rescale=1./255)
# train_generator = train_datagen.flow_from_directory("dataset/train/", batch_size=128,
#                                                     class_mode='binary', shuffle=True, target_size=(img_width,img_height))
# test_generator = test_datagen.flow_from_directory("dataset/test/",
#                                                   batch_size=128,
#                                                   class_mode='binary',
#                                                   shuffle=True,
#                                                   target_size=(img_width,img_height))
# train_steps = train_generator.n // train_generator.batch_size  #  to determine which numbers of batch_size in epoch
# test_steps = test_generator.n // test_generator.batch_size
#
# print(train_steps)
# print(test_steps)
#
# ###### Define a Model
# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3), padding='same'))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# #
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# #
# model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# #
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(1, activation='sigmoid'))
#
# ##
# model.summary()
# model.compile(loss='binary_crossentropy', optimizer=Adam(0.005), metrics=['acc'])
# #### Train Model
# checkpointer = ModelCheckpoint(filepath='weights_custom_model.hdf5', verbose=1)
# history = model.fit(train_generator,
#                     steps_per_epoch=train_steps,
#                     epochs=100,
#                     validation_data=test_generator,
#                     validation_steps=test_steps,
#                     verbose=1, callbacks=[checkpointer])
#
# import matplotlib.pyplot as plt
# accuracy_train = history.history['acc']
# val_accuracy = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(len(accuracy_train))
#
# plt.plot(epochs, accuracy_train)
# plt.plot(epochs, val_accuracy)
# plt.title('Training and validation accuracy')
# plt.figure()
#
# # Plot training and validation loss per epoch
# plt.plot(epochs, loss)
# plt.plot(epochs, val_loss)
# plt.title('Training and validation loss')
# plt.show()
# print(model.evaluate(test_generator, verbose=1, steps=test_steps))
# # ############## output
# # ##  100s 7s/step - loss: 0.6933 - acc: 0.4808 - val_loss: 0.6929 - val_acc: 0.5599
# #
# ####################### Saving the model
# # model.save('model.h5')
# #
from tensorflow.keras.models import load_model

model = load_model('model.h5')

################### for test single example
from tensorflow.keras.preprocessing import image
import numpy as np
image_width, image_height = 150, 150
img = image.load_img("dataset/test/hot_dog/211824.jpg", target_size=(image_width, image_height))
img = image.img_to_array(img)   # to convert image to numpy array
img = np.expand_dims(img, axis=0)
img = img/255.
prediction = model.predict_classes(img).tolist()
print(prediction[0][0])