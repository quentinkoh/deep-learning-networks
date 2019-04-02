# import the necessary packages
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense,\
    BatchNormalization, Dropout


class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        '''
        Args:
            width: int, width of input image
            height: int, height of input image
            depth: int, depth of input image
            classes: int, number of class labels

        Returns:
            model: the network architecture of MiniVGGNet

        Notes:
            the minivggnet architecture can be summarized as follows:
            Input =>
            Conv => ReLU => BN =>
            Conv => ReLU => BN =>
            Pool => Dropout =>

            Conv => ReLU => BN =>
            Conv => ReLU => BN =>
            Pool => Dropout =>

            FC => ReLU => BN => Dropout => FC => Softmax
        '''
        model = Sequential()
        input_shape = (height, width, depth)
        channel_dim = -1

        # however, if keras backend is set to channels first, then we shall
        # update the input shape
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)
            channel_dim = 1

        # first set of layering
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dim))

        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dim))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # second set of layering
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dim))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dim))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # third set of layering
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model
