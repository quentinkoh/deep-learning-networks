# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Flatten, Dense
from keras import backend as K


class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        '''
        Args:
            width: int, width of input image
            height: int, height of input image
            depth: int, depth of input image
            classes: int, number of class labels

        Returns:
            model: the network architecture of ShallowNet

        Notes:
            the shallownet architecture can be summarized as:
            Input => Conv => ReLU => FC => Softmax
        '''
        # building the shallownet network architecture (default: channels last)
        model = Sequential()
        input_shape = (height, width, depth)

        # however, if keras backend is set to channels first, then we shall
        # update the input shape
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)

        # layering
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model
