import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend
import cv2 as cv
from keras.utils import np_utils
backend.set_image_dim_ordering("th")
r_dir="gesture/"
r_g=os.listdir(r_dir)
p_type=[]
num_all=0
for i in range(5):
    p_type.append(os.listdir(r_dir+r_g[i]))
    num_all+=len(p_type[i])

label=np.ones((num_all,),dtype=int)

sta=0
end_=len(p_type[0])
for i in range(5):
    label[sta:end_]=i
    if i ==4:
        break
    sta=end_
    end_+=len(p_type[i+1])
image_array=[]

for i in range(5):
    image_array.append([np.array(cv.imread(r_dir+r_g[i]+'/'+tmp,cv.IMREAD_GRAYSCALE)).flatten() for tmp in p_type[i]])

image_array=np.array(image_array,dtype='f').reshape(num_all,-1)

data,Label=shuffle(image_array,label,random_state=2)
train_data = [data, Label]
(X, y) = (train_data[0], train_data[1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

X_train=X_train.reshape(-1,1,200,200)
X_test=X_test.reshape(-1,1,200,200)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, num_classes=5)
Y_test = np_utils.to_categorical(y_test, num_classes=5)

model = Sequential()
model.add(Conv2D(32, 3,padding='valid',input_shape=(1, 200, 200)))
convout1 = Activation('relu')
model.add(convout1)
model.add(Conv2D(32, (3, 3)))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.summary()
model.fit(X_train, Y_train, batch_size=32, epochs=15,
                 verbose=1, validation_split=0.2)
model.save("gesture_model.h5")

loss, accuracy = model.evaluate(X_test, Y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)
