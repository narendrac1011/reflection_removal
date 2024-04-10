# reflection_removal
This repository is for the Computer Graphics and Image Preprocessing Final Project

## About the Project:
There are numerous number of ways to remove reflection from an image. I have implemented some of the techniques which you can use to remove reflections. Some of them are not so efficient but they somewhat remove the reflections.

1. AVERAGING TECHNIQUE: Averaging is a simple technique where a set of images are averaged pixel-wise to reduce the effects of reflections in that image. So you would need a set of images to perform this technique. We take the average of each pixel in the set of photos and display a new image based on the average pixel values.

Run: python averaging.py -i Image_Set_1 or python averaging.py -i Image_Set_2

2. 