# Original Repository

https://github.com/yhw-yhw/TalkSHOW

## Changes on Original Repository

Added an export_motions.py file under the scripts directory. This file can generate SMPLX motion data and save them as npz or pkls under visualise/poses directory. The npz or pkls can be further imported into Blender with the smplx_blender_addon script from https://talkshow.is.tue.mpg.de/. Yet the original smplx_blender_addon script contains some bugs, you cannot correctly import the SMPLX model directly. I modified the smplx_blender_addon script as well to make it work.


