from keras.layers import Input, Dense, Dropout, Activation, Concatenate, BatchNormalization
from keras.models import Model
from keras.layers import Conv2D, GlobalAveragePooling2D, AveragePooling2D
from keras.regularizers import l2


def unet(learning_rate, pretrained_weights=None, input_size=(1024, 1024, 1), down_sampling=4):
    """Almost directly taken from https://github.com/zhixuhao/unet. Modified to fit into memory"""
    inputs = klayer.Input(input_size)
    # Rescale to not take too much memory
    scaled_inputs = klayer.MaxPooling2D(pool_size=(down_sampling, down_sampling))(inputs)
    conv1 = klayer.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(scaled_inputs)
    conv1 = klayer.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = klayer.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = klayer.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = klayer.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = klayer.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = klayer.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = klayer.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = klayer.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = klayer.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = klayer.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = klayer.Dropout(0.5)(conv4)
    pool4 = klayer.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = klayer.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = klayer.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = klayer.Dropout(0.5)(conv5)

    up6 = klayer.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        klayer.UpSampling2D(size=(2, 2))(drop5))
    merge6 = klayer.concatenate([drop4, up6], axis=3)
    conv6 = klayer.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = klayer.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = klayer.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        klayer.UpSampling2D(size=(2, 2))(conv6))
    merge7 = klayer.concatenate([conv3, up7], axis=3)
    conv7 = klayer.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = klayer.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = klayer.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        klayer.UpSampling2D(size=(2, 2))(conv7))
    merge8 = klayer.concatenate([conv2, up8], axis=3)
    conv8 = klayer.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = klayer.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = klayer.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        klayer.UpSampling2D(size=(2, 2))(conv8))
    merge9 = klayer.concatenate([conv1, up9], axis=3)
    conv9 = klayer.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = klayer.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = klayer.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = klayer.Conv2D(1, 1, activation='sigmoid')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=klayer.UpSampling2D(size=(down_sampling, down_sampling))(conv10))

    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss=keras.losses.MSE, metrics=["accuracy"])

    # model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def smet(learning_rate, pretrained_weights=None, input_size=(1024, 1024, 1), down_sampling=4):
    """Almost directly taken from https://github.com/zhixuhao/unet. Modified to fit into memory"""
    inputs = klayer.Input(input_size)
    # Rescale to not take too much memory
    scaled_inputs = klayer.MaxPooling2D(pool_size=(down_sampling, down_sampling))(inputs)
    conv1 = klayer.Conv2D(4, 5, activation='tanh', padding='same', kernel_initializer='he_normal')(scaled_inputs)
    conv1 = klayer.Conv2D(1, 5, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv1)
    model = tf.keras.Model(inputs=inputs, outputs=klayer.UpSampling2D(size=(down_sampling, down_sampling))(conv1))

    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss=keras.losses.binary_crossentropy, metrics=["accuracy"])

    # model.summary()
    #all_layers_output = keras.backend.function([model.layers[0].input],
    #           [l.output for l in model.layers[1:]])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model#, all_layers_output

def DenseNet(input_shape=None, dense_blocks=3, dense_layers=-1, growth_rate=12, nb_classes=None, dropout_rate=None,
             bottleneck=False, compression=1.0, weight_decay=1e-4, depth=40):
    """
    Creating a DenseNet

    Arguments:
        input_shape  : shape of the input images. E.g. (28,28,1) for MNIST
        dense_blocks : amount of dense blocks that will be created (default: 3)
        dense_layers : number of layers in each dense block. You can also use a list for numbers of layers [2,4,3]
                       or define only 2 to add 2 layers at all dense blocks. -1 means that dense_layers will be calculated
                       by the given depth (default: -1)
        growth_rate  : number of filters to add per dense block (default: 12)
        nb_classes   : number of classes
        dropout_rate : defines the dropout rate that is accomplished after each conv layer (except the first one).
                       In the paper the authors recommend a dropout of 0.2 (default: None)
        bottleneck   : (True / False) if true it will be added in convolution block (default: False)
        compression  : reduce the number of feature-maps at transition layer. In the paper the authors recomment a compression
                       of 0.5 (default: 1.0 - will have no compression effect)
        weight_decay : weight decay of L2 regularization on weights (default: 1e-4)
        depth        : number or layers (default: 40)

    Returns:
        Model        : A Keras model instance
    """

    if nb_classes == None:
        raise Exception('Please define number of classes (e.g. num_classes=10). This is required for final softmax.')

    if compression <= 0.0 or compression > 1.0:
        raise Exception(
            'Compression have to be a value between 0.0 and 1.0. If you set compression to 1.0 it will be turn off.')

    if type(dense_layers) is list:
        if len(dense_layers) != dense_blocks:
            raise AssertionError('Number of dense blocks have to be same length to specified layers')
    elif dense_layers == -1:
        if bottleneck:
            dense_layers = (depth - (dense_blocks + 1)) / dense_blocks // 2
        else:
            dense_layers = (depth - (dense_blocks + 1)) // dense_blocks
        dense_layers = [int(dense_layers) for _ in range(dense_blocks)]
    else:
        dense_layers = [int(dense_layers) for _ in range(dense_blocks)]

    img_input = Input(shape=input_shape)
    nb_channels = growth_rate * 2

    print('Creating DenseNet')
    print('#############################################')
    print('Dense blocks: %s' % dense_blocks)
    print('Layers per dense block: %s' % dense_layers)
    print('#############################################')

    # Initial convolution layer
    x = Conv2D(nb_channels, (3, 3), padding='same', strides=(1, 1),
               use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)

    # Building dense blocks
    for block in range(dense_blocks):

        # Add dense block
        x, nb_channels = dense_block(x, dense_layers[block], nb_channels, growth_rate, dropout_rate, bottleneck,
                                     weight_decay)

        if block < dense_blocks - 1:  # if it's not the last dense block
            # Add transition_block
            x = transition_layer(x, nb_channels, dropout_rate, compression, weight_decay)
            nb_channels = int(nb_channels * compression)

    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(nb_classes, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(
        x)

    model_name = None
    if growth_rate >= 36:
        model_name = 'widedense'
    else:
        model_name = 'dense'

    if bottleneck:
        model_name = model_name + 'b'

    if compression < 1.0:
        model_name = model_name + 'c'

    return Model(img_input, x, name=model_name), model_name


def dense_block(x, nb_layers, nb_channels, growth_rate, dropout_rate=None, bottleneck=False, weight_decay=1e-4):
    """
    Creates a dense block and concatenates inputs
    """

    x_list = [x]
    for i in range(nb_layers):
        cb = convolution_block(x, growth_rate, dropout_rate, bottleneck, weight_decay)
        x_list.append(cb)
        x = Concatenate(axis=-1)(x_list)
        nb_channels += growth_rate
    return x, nb_channels


def convolution_block(x, nb_channels, dropout_rate=None, bottleneck=False, weight_decay=1e-4):
    """
    Creates a convolution block consisting of BN-ReLU-Conv.
    Optional: bottleneck, dropout
    """

    # Bottleneck
    if bottleneck:
        bottleneckWidth = 4
        x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = Conv2D(nb_channels * bottleneckWidth, (1, 1), use_bias=False, kernel_regularizer=l2(weight_decay))(x)
        # Dropout
        if dropout_rate:
            x = Dropout(dropout_rate)(x)

    # Standard (BN-ReLU-Conv)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_channels, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)

    # Dropout
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_layer(x, nb_channels, dropout_rate=None, compression=1.0, weight_decay=1e-4):
    """
    Creates a transition layer between dense blocks as transition, which do convolution and pooling.
    Works as downsampling.
    """

    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(int(nb_channels * compression), (1, 1), padding='same',
               use_bias=False, kernel_regularizer=l2(weight_decay))(x)

    # Adding dropout
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x