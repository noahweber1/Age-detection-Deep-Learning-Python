# import packages
import os
import random
from scipy.misc import imresize
import pandas as pd
from scipy.misc import imread
import keras
from sklearn.preprocessing import LabelEncode
from keras.models import Sequential
from keras.layers import Dense, Flatten, InputLayer


# load the files via operating system commands, alternatively directly, important is to have training (70%) and test (30%) set

root_dir = os.path.abspath('.')
data_dir = '/mnt/hdd/datasets/misc'

train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

# let us resize the images so that we can feed them to network
temp = []
for img_name in train.ID:
    img_path = os.path.join(data_dir, 'Train', img_name)
    img = imread(img_path)
    img = imresize(img, (32, 32))
    img = img.astype('float32') # this will help us in later stage
    temp.append(img)

train_x = np.stack(temp)

# same for test sets

temp = []
for img_name in test.ID:
    img_path = os.path.join(data_dir, 'Test', img_name)
    img = imread(img_path)
    img = imresize(img, (32, 32))
    temp.append(img.astype('float32'))

test_x = np.stack(temp)

# since NN only accepts numerical values as input let us encode them as one-hot vectors


test_x = np.stack(temp) # like concatenate, joining arrays
lb = LabelEncoder()
train_y = lb.fit_transform(train.Class)
train_y = keras.utils.np_utils.to_categorical(train_y)

# now we are ready to feed the network, just a couple of specifcations

from keras.layers import Dense, Flatten, InputLayer


input_num_units = (32, 32, 3)
hidden_num_units = 500
output_num_units = 3

epochs = 5
batch_size = 128


model = Sequential([
  InputLayer(input_shape=input_num_units),
  Flatten(),
  Dense(units=hidden_num_units, activation='relu'),
  Dense(units=output_num_units, activation='softmax'),
])



# compile and train the network

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=batch_size,epochs=epochs,verbose=1)


# we need to also validate if we want to make sure that model performs well on both the data it is training on and on a new testing data

vmodel.fit(train_x, train_y, batch_size=batch_size,epochs=epochs,verbose=1, validation_split=0.2)

pred = model.predict_classes(test_x)
pred = lb.inverse_transform(pred)

test['Class'] = pred


# let us inspect visually how our model performs (we now that there are 3 classes that we can predict MIDDLE, OLD, YOUNG aged)

i = random.choice(train.index)
img_name = train.ID[i]

img = imread(os.path.join(data_dir, 'Train', img_name)).astype('float32')
imshow(imresize(img, (128, 128)))
pred = model.predict_classes(train_x)
print('Original:', train.Class[i], 'Predicted:', lb.inverse_transform(pred[i]))

Original: MIDDLE Predicted: MIDDLE
