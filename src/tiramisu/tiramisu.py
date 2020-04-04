"""The standard Tiramisu model."""
from keras.layers import Activation
from keras.models import Model
from keras.optimizers import RMSprop
from ..losses import build_categorical_crossentropy
from ..metrics import build_categorical_accuracy
from ._core import build_tiramisu


def tiramisu(image_shape: tuple, num_classes: int,
    class_weights=None,
    initial_filters: int=48,
    growth_rate: int=16,
    layer_sizes: list=[4, 5, 7, 10, 12],
    bottleneck_size: int=15,
    dropout: float=0.2,
    learning_rate: float=1e-3,
    mc_dropout=False,
) -> Model:
    """
    Build a Tiramisu model for the given image shape.

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
        mc_dropout: whether to use Monte Carlo dropout at test time

    Returns:
        a compiled model of the Tiramisu architecture

    """
    # build the base of the network
    inputs, logits = build_tiramisu(image_shape, num_classes,
        initial_filters=initial_filters,
        growth_rate=growth_rate,
        layer_sizes=layer_sizes,
        bottleneck_size=bottleneck_size,
        dropout=dropout,
        mc_dropout=mc_dropout,
    )
    # pass the logits through the Softmax activation to get probabilities
    softmax = Activation('softmax', name='softmax')(logits)
    # build the Tiramisu model
    model = Model(inputs=[inputs], outputs=[softmax])
    # compile the model
    model.compile(
        optimizer=RMSprop(lr=learning_rate),
        loss=build_categorical_crossentropy(class_weights),
        metrics=[build_categorical_accuracy(weights=class_weights)],
    )

    return model


# explicitly define the outward facing API of this module
__all__ = [tiramisu.__name__]
