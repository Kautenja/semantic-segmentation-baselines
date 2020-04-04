"""Tiramisu model that has head split to estimate aleatoric uncertainty."""
from keras.layers import Activation
from keras.layers import Conv2D
from keras.models import Model
from keras.optimizers import RMSprop
from ..layers import Stack
from ..losses import build_categorical_crossentropy
from ..losses import build_categorical_aleatoric_loss
from ..metrics import build_categorical_accuracy
from ._core import build_tiramisu


def aleatoric_tiramisu(image_shape: tuple, num_classes: int,
    class_weights=None,
    initial_filters: int=48,
    growth_rate: int=16,
    layer_sizes: list=[4, 5, 7, 10, 12],
    bottleneck_size: int=15,
    dropout: float=0.2,
    learning_rate: float=1e-3,
    samples: int=50,
    weights_file: str=None,
    freeze_weights: bool=True,
):
    """
    Build a Tiramisu model that computes Aleatoric uncertainty.

    Args:
        image_shape: the image shape to create the model for
        num_classes: the number of classes to segment for (e.g. c)
        class_weights: the weights for each class
        initial_filters: the number of filters in the first convolution layer
        growth_rate: the growth rate to use for the network (e.g. k)
        layer_sizes: a list with the size of each dense down-sample block.
                     reversed to determine the size of the up-sample blocks
        bottleneck_size: the number of convolutional layers in the bottleneck
        dropout: the dropout rate to use in dropout layers
        learning_rate: the learning rate for the RMSprop optimizer
        samples: the number of samples for Monte Carlo loss estimation
        weights_file: base Tiramisu network weights to train from
        freeze_weights: whether to freeze all other weights

    Returns:
        a compiled model of the Tiramisu architecture + Aleatoric

    """
    # build the base of the network
    inputs, logits, sigma = build_tiramisu(image_shape, num_classes,
        initial_filters=initial_filters,
        growth_rate=growth_rate,
        layer_sizes=layer_sizes,
        bottleneck_size=bottleneck_size,
        dropout=dropout,
        split_head=True,
    )
    # load the pre-trained weights only if a file is provided
    if weights_file is not None:
        # pass the logits through the Softmax activation
        softmax = Activation('softmax', name='softmax')(logits)
        # build the standard Tiramisu model
        tiramisu = Model(inputs=[inputs], outputs=[softmax])
        # load the weights file for the model
        tiramisu.load_weights(weights_file)

    # stack the logits and sigma for aleatoric loss
    aleatoric = Stack(name='aleatoric')([logits, sigma])
    # pass the logits through the Softmax activation to get probabilities
    softmax = Activation('softmax', name='softmax')(logits)
    # build the Tiramisu model
    model = Model(inputs=[inputs], outputs=[softmax, sigma, aleatoric])

    # iterate over the layers in the network
    for layer in model.layers:
        # skip the convolutional layer to estimate aleatoric
        if layer.name == 'sigma':
            continue
        # set the trainable state of all other layers
        layer.trainable = not freeze_weights

    # compile the model
    model.compile(
        optimizer=RMSprop(lr=learning_rate),
        loss={
            'softmax': build_categorical_crossentropy(class_weights),
            'aleatoric': build_categorical_aleatoric_loss(samples)
        },
        metrics={'softmax': [build_categorical_accuracy(weights=class_weights)]},
    )

    return model


# explicitly define the outward facing API of this module
__all__ = [aleatoric_tiramisu.__name__]
