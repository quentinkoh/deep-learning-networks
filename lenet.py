# import the necessary packages
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense


class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        '''
        Args:
            width: int, width of input image
            height: int, height of input image
            depth: int, depth of input image
            classes: int, number of class labels

        Returns:
            model: the network architecture of LeNet

        Notes:
            the lenet architecture can be summarized as follows:
            Input = > Conv = > ReLU => Pool => Conv => ReLU => Pool
            => FC => ReLU => FC => Softmax
        '''
        model = Sequential()
        input_shape = (height, width, depth)

        # however, if keras backend is set to channels first, then we shall
        # update the input shape
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)

        # first set of Conv => ReLU => Pool layer
        model.add(Conv2D(20, (5, 5), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of Conv => ReLU => Pool layer
        model.add(Conv2D(50, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # FC => ReLU => FC
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))

        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model
