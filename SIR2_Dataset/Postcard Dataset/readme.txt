This set contains all the postcard dataset

In the postcard dataset, 

ab is the sequence number for the corresponding data. 

Focus file means that the varying parameters in this file is the aperature size of the camera. 

It leads to the change of the Depth of field of the camera and the blur levels of the reflection in the final captured image. 

11, 13, 16, 19 22, 27, 32 are the aperature sizes we use to capture these images. 

Thickness files means that the varying parameters in this file is the thickness of the glass. 

It leads to different ghosting effects of the reflections. 3, 5, 10 is the glass thickness we use
to capture these images. In the images taken with thicker glass, you can oberver more obvious spatial shift of the reflections. 

When you look at the images in each file, if their names contain 'g', it means that they are the groundtruth of the background
If their names contain 'r', it means taht they are the groundtruth for the reflection. 
If their names contain 'm', it means that they are the mixture images. 

Due to the refractive effect of the glass, the mixture images and the ground truth for background have some spatial shifts in the original images taken by the camera. 
To reduce the refractive effect caused by the glass when taking photos, we use some image alignment methods to register images.

This works well in the postcard dataset. 

The postcard dataset is maily designed for exploring the influence of different parameters.