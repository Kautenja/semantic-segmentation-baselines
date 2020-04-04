"""Methods to get unwrapped predictions from different architectures."""
import numpy as np
from .utils import extract_aleatoric
from .utils import heatmap


def predict(model, generator, camvid):
    """
    Return post-processed predictions for the given generator.

    Args:
        model: the model to use to predict with
        generator: the generator to get data from
        camvid: the CamVid instance for un-mapping target values

    Returns:
        a tuple of for NumPy tensors with RGB data:
        - the batch of RGB X values
        - the unmapped RGB batch of y values
        - the unmapped RGB predicted values from the model

    """
    if isinstance(generator, np.ndarray):
        # get predictions from the model
        y_pred = model.predict(generator)
        # return a tuple of RGB pixel data
        return generator, camvid.unmap(y_pred)

    # generate a batch of data from the generator
    imgs, y_true = next(generator)
    # get predictions from the model
    y_pred = model.predict(imgs)
    # return a tuple of RGB pixel data
    return imgs, camvid.unmap(y_true), camvid.unmap(y_pred)


def predict_aleatoric(model, generator, camvid) -> tuple:
    """
    Return post-processed predictions for the given generator.

    Args:
        model: the aleatoric model to use to predict with
        generator: the generator to get data from
        camvid: the CamVid instance for un-mapping target values

    Returns:
        a tuple of for NumPy tensors with RGB data:
        - the batch of RGB X values
        - the unmapped RGB batch of y values
        - the unmapped RGB predicted values from the model
        - the heat-map RGB values of the aleatoric uncertainty

    """
    if isinstance(generator, np.ndarray):
        # predict mean values and variance
        y_pred, sigma, _ = model.predict(generator)
        # extract the aleatoric uncertainty from the tensor
        sigma2 = extract_aleatoric(sigma, y_pred)
        # return X values, unmapped y and u values, and heat-map of s2
        return generator, camvid.unmap(y_pred), heatmap(sigma2)

    # get the batch of data
    imgs, y_true = next(generator)
    # predict mean values and variance
    y_pred, sigma, _ = model.predict(imgs)
    # extract the aleatoric uncertainty from the tensor
    sigma2 = extract_aleatoric(sigma, y_pred)
    # return X values, unmapped y and u values, and heat-map of s2
    return imgs, camvid.unmap(y_true[0]), camvid.unmap(y_pred), heatmap(sigma2)


def predict_epistemic(model, generator, camvid) -> tuple:
    """
    Return post-processed predictions for the given generator.

    Args:
        model: the epistemic model to use to predict with
        generator: the generator to get data from
        camvid: the CamVid instance for un-mapping target values

    Returns:
        a tuple of for NumPy tensors with RGB data:
        - the batch of RGB X values
        - the unmapped RGB batch of y values
        - the unmapped RGB predicted mean values from the model
        - the heat-map RGB values of the epistemic uncertainty

    """
    if isinstance(generator, np.ndarray):
        # predict mean values and variance
        y_pred, sigma2 = model.predict(generator)
        # return X values, unmapped y and u values, and heat-map of sigma**2
        return generator, camvid.unmap(y_pred), heatmap(sigma2)

    # get the batch of data
    imgs, y_true = next(generator)
    # predict mean values and variance
    y_pred, sigma2 = model.predict(imgs)
    # return X values, unmapped y and u values, and heat-map of sigma**2
    return imgs, camvid.unmap(y_true), camvid.unmap(y_pred), heatmap(sigma2)


def predict_hyrbid(model, generator, camvid) -> tuple:
    """
    Return post-processed predictions for the given generator.

    Args:
        model: the hybrid model to use to predict with
        generator: the generator to get data from
        camvid: the CamVid instance for un-mapping target values

    Returns:
        a tuple of for NumPy tensors with RGB data:
        - the batch of RGB X values
        - the unmapped RGB batch of y values
        - the unmapped RGB predicted mean values from the model
        - the heat-map RGB values of the epistemic uncertainty
        - the heat-map RGB values of the aleatoric uncertainty

    """
    if isinstance(generator, np.ndarray):
        # predict mean values and variance
        y_pred, aleatoric, epistemic = model.predict(generator)
        # extract the aleatoric uncertainty from the tensor
        aleatoric = extract_aleatoric(aleatoric, y_pred)
        # return X values, unmapped y and u values, and heat-map of sigma**2
        return (
            generator,
            camvid.unmap(y_pred),
            heatmap(epistemic),
            heatmap(aleatoric),
        )

    # get the batch of data
    imgs, y_true = next(generator)
    # predict mean values and variance
    y_pred, aleatoric, epistemic = model.predict(imgs)
    # extract the aleatoric uncertainty from the tensor
    aleatoric = extract_aleatoric(aleatoric, y_pred)
    # return X values, unmapped y and u values, and heat-map of sigma**2
    return (
        imgs,
        camvid.unmap(y_true),
        camvid.unmap(y_pred),
        heatmap(epistemic),
        heatmap(aleatoric),
    )


# explicitly define the outward facing API of this module
__all__ = [
    predict.__name__,
    predict_aleatoric.__name__,
    predict_epistemic.__name__,
    predict_hyrbid.__name__,
]
