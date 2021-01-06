from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Dense,
    Flatten,
    Dropout,
    Add,
    Activation,
)

from tensorflow.keras.optimizers import (
    SGD, Adam, RMSprop
)





def model_vgg16():
    model = Sequential()
    model.add(Conv2D(input_shape=(tgt_size,tgt_size,3),filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))


    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(units=4096,activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(units=1000,activation="linear"))
    
    # len(class)
    model.add(Dense(units=4, activation="softmax"))

    opt = SGD(lr=0.00001, momentum=0.9, decay=0.0005)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    return model
    
    
model = model_vgg16()
