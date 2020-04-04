"""A method to predict a video of frames using segmentation models."""
import imageio
import numpy as np
from tqdm import tqdm
from skimage import transform


def predict_video(
    video_path: str,
    out_path: str,
    camvid: 'CamVid',
    model: 'Model',
    predict: 'Callable',
) -> None:
    """
    Predict a video stream and stream the predictions to disk.

    Args:
        video_path: the path to the video to stream
        out_path: the path to write the output video stream to
        camvid: the CamVid instance for un-mapping segmentations
        model: the model to generate predictions and uncertainty from
        predict: the predict method for the given model

    Returns:
        None

    """
    # create a stream to read the input video
    reader = imageio.get_reader(video_path)
    # the shape to resize frames to
    image_shape = model.input_shape[1:]
    # create a video writer with source FPS
    writer = imageio.get_writer(out_path, fps=reader.get_meta_data()['fps'])

    # iterate over the frames in the source video
    for frame in tqdm(reader):
        # resize the image to the acceptable size for the model
        img = transform.resize(frame, image_shape,
            anti_aliasing=False,
            mode='symmetric',
            clip=False,
            preserve_range=True,
        )[None, ...]
        # predict mean and model variance of the frame
        outs = predict(model, img, camvid)
        height, width, channels = image_shape
        # convert the three images into a singular image (side-by-side)
        image = np.zeros((height, width * len(outs), channels), dtype='uint8')
        for idx, piece in enumerate(outs):
            image[:, idx * width:(idx + 1) * width, :] = piece
        # save the image to the stream
        writer.append_data(image)

    # close the writer
    writer.close()
    # close the reader
    reader.close()


# explicitly define the outward facing API of this module
__all__ = [predict_video.__name__]
