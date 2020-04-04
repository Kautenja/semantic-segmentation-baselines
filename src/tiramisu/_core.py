"""Core methods for building Tiramisu networks."""
from keras import backend as K
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.regularizers import l2


# static arguments used for all convolution layers in Tiramisu models
CONV = dict(
    kernel_initializer='he_uniform',
    kernel_regularizer=l2(1e-4),
)


def dense_block(inputs,
    num_layers: int,
    num_filters: int,
    skip=None,
    dropout: float=0.2,
    mc_dropout: bool=None,
):
    """
    Create a dense block for a given input tensor.

    Args:
        inputs: the input tensor to append this dense block to
        num_layers: the number of layers in this dense block
        num_filters: the number of filters in the convolutional layer
        skip: the skip mode of the dense block as a {str, None, Tensor}
            - 'downstream': the dense block is part of the down-sample side
            - None: the dense block is the bottleneck block bottleneck
            - a skip tensor: the dense block is part of the up-sample side
        dropout: the dropout rate to use per layer (None to disable dropout)
        mc_dropout: whether to use dropout in test (True) time or not (None)

    Returns:
        a tensor with a new dense block appended to it

    """
    # create a placeholder list to store references to output tensors
    outputs = [None] * num_layers
    # if skip is a tensor, concatenate with inputs (upstream mode)
    if K.is_tensor(skip):
        # concatenate the skip with the inputs
        inputs = Concatenate()([inputs, skip])
    # copy a reference to the block inputs for later
    block_inputs = inputs
    # iterate over the number of layers in the block
    for idx in range(num_layers):
        # training=True to compute current batch statistics during inference
        # i.e., during training, validation, and testing
        x = BatchNormalization()(inputs, training=True)
        x = Activation('relu')(x)
        x = Conv2D(num_filters, kernel_size=(3, 3), padding='same', **CONV)(x)
        if dropout is not None:
            x = Dropout(dropout)(x, training=mc_dropout)
        # store the output tensor of this block to concatenate at the end
        outputs[idx] = x
        # concatenate the input with the outputs (unless last layer in block)
        if idx + 1 < num_layers:
            inputs = Concatenate()([inputs, x])

    # concatenate outputs to produce num_layers * num_filters feature maps
    x = Concatenate()(outputs)
    # if skip is 'downstream' concatenate inputs with outputs (downstream mode)
    if skip == 'downstream':
        x = Concatenate()([block_inputs, x])

    return x


def transition_down_layer(inputs, dropout: float=0.2, mc_dropout: bool=None):
    """
    Create a transition layer for a given input tensor.

    Args:
        inputs: the input tensor to append this transition down layer to
        dropout: the dropout rate to use per layer (None to disable dropout)
        mc_dropout: whether to use dropout in test (True) time or not (None)

    Returns:
        a tensor with a new transition down layer appended to it

    """
    # get the number of filters from the input activation maps
    num_filters = K.int_shape(inputs)[-1]
    # training=True to compute current batch statistics during inference
    # i.e., during training, validation, and testing
    x = BatchNormalization()(inputs, training=True)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, kernel_size=(1, 1), padding='same', **CONV)(x)
    if dropout is not None:
        x = Dropout(dropout)(x, training=mc_dropout)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    return x


def transition_up_layer(inputs):
    """
    Create a transition up layer for a given input tensor.

    Args:
        inputs: the input tensor to append this transition up layer to

    Returns:
        a tensor with a new transition up layer appended to it

    """
    # get the number of filters from the number of activation maps
    return Conv2DTranspose(K.int_shape(inputs)[-1],
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='same',
        **CONV
    )(inputs)


def build_tiramisu(image_shape: tuple, num_classes: int,
    initial_filters: int=48,
    growth_rate: int=16,
    layer_sizes: list=[4, 5, 7, 10, 12],
    bottleneck_size: int=15,
    dropout: float=0.2,
    mc_dropout: bool=None,
    split_head: bool=False,
) -> Model:
    """
    Build a Tiramisu model for the given image shape.

    Args:
        image_shape: the image shape to create the model for
        num_classes: the number of classes to segment for (e.g. c)
        initial_filters: the number of filters in the first convolution layer
        growth_rate: the growth rate to use for the network (e.g. k)
        layer_sizes: a list with the size of each dense down-sample block.
                     reversed to determine the size of the up-sample blocks
        bottleneck_size: the number of convolutional layers in the bottleneck
        dropout: the dropout rate to use in dropout layers
        mc_dropout: whether to use dropout in test (True) time or not (None)
        split_head: whether to split the head of the network (i.e., 2 outputs)

    Returns:
        a tuple of
        - the input layer
        - the logits output
        - the sigma output if split_head is True

    """
    # ensure the image shape is legal for the architecture
    div = int(2**len(layer_sizes))
    for dim in image_shape[:-1]:
        # raise error if the dimension doesn't evenly divide
        if dim % div:
            msg = 'dimension ({}) must be divisible by {}'.format(dim, div)
            raise ValueError(msg)
    # the input block of the network
    inputs = Input(image_shape, name='Tiramisu_input')
    # assume 8-bit inputs and convert to floats in [0,1]
    x = Lambda(lambda x: x / 255.0, name='pixel_norm')(inputs)
    # the initial convolution layer
    x = Conv2D(initial_filters, kernel_size=(3, 3), padding='same', **CONV)(x)
    # the down-sampling side of the network (keep outputs for skips)
    skips = [None] * len(layer_sizes)
    # iterate over the size for each down-sampling block
    for idx, size in enumerate(layer_sizes):
        skips[idx] = dense_block(x, size, growth_rate,
            skip='downstream',
            dropout=dropout,
            mc_dropout=mc_dropout
        )
        x = transition_down_layer(skips[idx],
            dropout=dropout,
            mc_dropout=mc_dropout
        )
    # the bottleneck of the network
    x = dense_block(x, bottleneck_size, growth_rate,
        dropout=dropout,
        mc_dropout=mc_dropout
    )
    # the up-sampling side of the network (using kept outputs for skips)
    for idx, size in reversed(list(enumerate(layer_sizes))):
        x = transition_up_layer(x)
        x = dense_block(x, size, growth_rate,
            skip=skips[idx],
            dropout=dropout,
            mc_dropout=mc_dropout
        )
    # the classification block
    head = lambda name: Conv2D(num_classes,
        kernel_size=(1, 1),
        padding='valid',
        name=name,
        **CONV
    )
    # calculate the logits of the network
    logits = head('logits')(x)
    # add the sigma layer if the head is split
    if split_head:
        sigma = head('sigma')(x)
        return inputs, logits, sigma

    return inputs, logits


# explicitly define the outward facing API of this module
__all__ = [build_tiramisu.__name__]
