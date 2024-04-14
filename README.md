# reflection_removal
This repository is for the Computer Graphics and Image Preprocessing Final Project

## About the Project:
There are numerous number of ways to remove reflection from an image. I have implemented some of the techniques which you can use to remove reflections. Some of them are not so efficient but they somewhat remove the reflections.

1. AVERAGING TECHNIQUE: Averaging is a simple technique where a set of images are averaged pixel-wise to reduce the effects of reflections in that image. So you would need a set of images to perform this technique. We take the average of each pixel in the set of photos and display a new image based on the average pixel values.

Run: python averaging.py -i Image_Set_1 or python averaging.py -i Image_Set_2

2. Independent Component Analysis (ICA): In the context of reflection removal, ICA can be applied to separate the mixed signal of a image into its constituent components, such as the reflection and the background scene. The input would be two images of a same scene which are captured at different polarizing orientations. One of them should contain the direct scene information, i.e. the background, and the other contains a combination of the direct scene information and the reflected component due to polarizing reflections.

After applying ICA, the resulting independent components represent different aspects of the scene. One component corresponds to the background scene without reflection, while another component represents the reflection itself. Once the reflection component is isolated, it can be subtracted or attenuated from the original image to obtain a reflection-free version of the scene.

This way we get rid of the reflection in images.

Run: 

Reference: Separating Reflections from Images Using Independent Components Analysis - https://dspace.mit.edu/bitstream/handle/1721.1/6675/AIM-1647.pdf?sequence


3. Using GANs: 