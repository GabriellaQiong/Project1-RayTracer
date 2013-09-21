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
   Raycasting from a camera into a scene through a pixel grid
   Phong lighting for one point light source
   Diffuse lambertian surfaces
   Raytraced shadows
   Cube intersection testing
   Sphere surface point sampling
   Specular reflection (not so good)
   Note: I will try to add the field of view feature soon

* Screen shots
Sorry, I will put the photo here soon. The screen shots have been included in the git.

* Video of project
It is in the git as well. I will put it in this markdown file soon. Sorry about that.

* Performance evaluation
I have tried to change the number of tiles to see whether the frame per second have changed accordingly.

* Some words must say
As an engineer in computer vision area, using GPU to do ray tracing with rendering is fantastic and totally new for me. Although the current result is not even good so far. I can say I have tried my best to learn and to do. I will also keep on improving it. Thanks to Patrick and Liam.

