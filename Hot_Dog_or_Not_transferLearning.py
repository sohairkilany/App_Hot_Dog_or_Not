######### Import Libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

####### Load Data
img_width, img_height = 150, 150
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    width_shift_range=0.4,
    height_shift_range=0.4,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory("dataset/train/", batch_size=128,
                                                    class_mode='binary', shuffle=True,
                                                    target_size=(img_width, img_height))
test_generator = test_datagen.flow_from_directory("dataset/test/",
                                                  batch_size=128,
                                                  class_mode='binary',
                                                  shuffle=True, target_size=(img_width,img_height))

train_steps = train_generator.n // train_generator.batch_size  #  to determine which numbers of batch_size in epoch
test_steps = test_generator.n // test_generator.batch_size

print(train_steps)
print(test_steps)
########
#print(train_generator.class_indices)
############### Fine Tuning with VGG Model
vgg_model = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(img_width, img_height, 3))
## to print 1000 classes
#vgg16 = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=True, input_tensor=None, input_shape=None, pooling=None, classes=1000)
#print(vgg16.decode_predictions(np.arange(1000), top=1000)

for layer in (vgg_model.layers):
    layer.trainable=False

model = Sequential()
# Add the vgg convolutional base model
model.add(vgg_model)
# Add new layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

model.summary()
model.summary()
model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001), metrics=['acc'])
#### Train Model
history = model.fit(train_generator,
                    steps_per_epoch=train_steps,
                    epochs=50,
                    validation_data=test_generator,
                    validation_steps=test_steps,
                    verbose=1)

import matplotlib.pyplot as plt
accuracy_train = history.history['acc']
val_accuracy = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy_train))

plt.plot(epochs, accuracy_train)
plt.plot(epochs, val_accuracy)
plt.title('Training and validation accuracy')
plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')
plt.show()
print(model.evaluate(test_generator, verbose=1, steps=test_steps))
########### 153s 10s/step - loss: 0.3678 - acc: 0.8365 - val_loss: 0.3670 - val_acc: 0.8346
############to save model
model.save('model_transfer.h5')
