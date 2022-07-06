from keras.layers import Input, Conv2D, MaxPooling2D
from keras import Model

inputs = Input((32, 32, 1))
x = Conv2D(32, (5, 5), strides=1, padding="valid")(inputs)
x = MaxPooling2D()(x)

model = Model(inputs=inputs, outputs=x)
model.summary()