"""Convert an MP4 file to a GIF file.

python make_gif.py <mp4 file>

"""
import sys
from moviepy.editor import VideoFileClip


try:
    # try to get the video file from the command line
    video_file = sys.argv[1]
except IndexError:
    # invalid input args, print documentation and exit
    print(__doc__)
    sys.exit(-1)


# load the input clip
video = VideoFileClip(video_file)
# clip the video into a range of ~20s
video = video.subclip((0, 50), (1, 10))
# use a small frame rate to keep file size low
video = video.set_fps(5)
# downscale the video by 2x to keep file size low
video = video.resize(0.5)
# write the clip as a GIF
video.write_gif(video_file.replace('.mp4', '.gif'))
