-------------------------------------------------------------------------------
CIS565: Project 1: CUDA Raytracer
-------------------------------------------------------------------------------
Fall 2013
-------------------------------------------------------------------------------
Due Thursday, 09/19/2013
-------------------------------------------------------------------------------
Qiong Wang
-------------------------------------------------------------------------------


* Description of the project

Considering each pixel in the image, we can find one ray from the camera. Along the ray, we can find the nearest object in the environment. And after determining whether the object is the light source, we can find specific color on the intersection and then pull it back on the pixel in the image. When computing the color of the intersection point, the diffusion, specular, ambient light are all considered for rendering the whole environment.


* Features implemented
 
   1. Raycasting from a camera into a scene through a pixel grid

   2. Phong lighting for one point light source
   
   3. Diffuse lambertian surfaces
   
   4. Raytraced shadows
   
   5. Cube intersection testing
   
   6. Sphere surface point sampling
   
   7. Specular reflection (optional)
   
   8. Soft shadows and area lights (optional)


* Screen shots

This is the screen shot when the code is running.

![ScreenShot](https://raw.github.com/GabriellaQiong/Project1-RayTracer/master/09222118snip.PNG)

This is the final render image.

![ScreenShot](https://raw.github.com/GabriellaQiong/Project1-RayTracer/master/09222119snip.PNG)


* Video of project

This is the video of the rendering process.

![[ScreenShot](https://raw.github.com/GabriellaQiong/Project1-RayTracer/master/09222119snip.PNG](http://www.youtube.com/watch?v=c3I9oAfzO8w)

The youtube link is here if you cannot open the video in the markdown file: http://www.youtube.com/watch?v=c3I9oAfzO8w


* Performance evaluation

Here is the table for the performance evaluation when changing the size of tile. We can easily find that when the tile size become larger the fps increases at the same time.

| tileSize  |     time for running 1000 iteration    |  approximate fps |
|:---------:|:--------------------------------------:|:----------------:|
|     1     |               2 : 22.7                 |       7.00       |
|     2     |               0 : 49.2                 |       20.33      |
|     4     |               0 : 17.4                 |       57.47      |
|     8     |               0 : 16.9                 |       59.17      |
|    16     |               0 : 16.5                 |       60.61      |



* Some words must say

As an engineer in computer vision area, using GPU to do ray tracing with rendering is fantastic and totally new for me. Although it was two days later than the deadline when I finished all these, I am so happy to get some good results. Thanks to Patrick and Liam.


* References

Ray Tracing Algorithm: http://cse.csusb.edu/tong/courses/cs621/notes/ray.php

Ray Tracing Pseudo-Codes: http://www.cs.unc.edu/~rademach/xroads-RT/RTarticle.html

Box Intersection: http://www.scratchapixel.com/lessons/3d-basic-lessons/lesson-7-intersecting-simple-shapes/ray-sphere-intersection/





