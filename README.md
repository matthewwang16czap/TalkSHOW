# Original Repository

https://github.com/yhw-yhw/TalkSHOW

I strongly recommend going through the original README.

## Changes on Original Repository

Added an export_motions.py file under the scripts directory. This file can generate SMPLX motion data and save them as npz or pkls under visualise/poses directory. To use this file, run it like other scripts.

The npz or pkls files can be further imported into Blender with the smplx_blender_addon script from https://talkshow.is.tue.mpg.de/. Yet the original smplx_blender_addon script by Joachim Tesch contains some bugs, you cannot correctly import the SMPLX model directly. You can either try to use projects like https://github.com/Meshcapade/SMPL_blender_addon, or fix it by yourself. Hints: I note the struct of the npz files in my code and it should be helpful to understand SMPLX model and make your own adjustments. 


