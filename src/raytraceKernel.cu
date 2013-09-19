// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
// Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
// Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
// Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
    exit(EXIT_FAILURE);
  }
}

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
  // Focal length in pixels
  float focal = resolution.y / 2.0f / tan(fov.y * (PI / 180)); 

  view = glm::normalize(view);
  up   = glm::normalize(up);
  glm::vec3 right   = glm::cross(view, up);

  // 3-D raycast vector from the camera in pixels
  glm::vec3 rayCast = focal * view + (x - resolution.x / 2.0f) * right - (y - resolution.y / 2.0f) * up;

  // Output the data
  ray r;
  r.origin = eye;
  r.direction = glm::normalize(rayCast);
  return r;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, material* materials, int numberOfMaterials, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, int* lights, int numberOfLights){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if((x<=resolution.x && y<=resolution.y)){
	  // Generate the ray
	  ray cameraRay = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);
	  
		// Initialize the object index, distance and the intersection point and the normal
	  int objectIndex = -1;
	  float updateDistance, distance = 1e8;
	  glm::vec3 updateIntersectionPoint, intersectionPoint;
	  glm::vec3 updateNormal, normal;

	  for(int i = 0; i < numberOfGeoms; ++ i){
		  glm::vec3 updateIntersectionPoint, updateNormal;
		  if(geoms[i].type == SPHERE){
		    updateDistance = sphereIntersectionTest(geoms[i], cameraRay, updateIntersectionPoint, updateNormal);
			if(updateDistance != -1 && updateDistance < distance){
			  distance = updateDistance;
			  intersectionPoint = updateIntersectionPoint;
			  normal = updateNormal;
			  objectIndex = i;
			}
		  }else if (geoms[i].type == CUBE){
			  updateDistance = boxIntersectionTest(geoms[i], cameraRay, updateIntersectionPoint, updateNormal);
			  if(updateDistance != -1 && updateDistance < distance){
				distance = updateDistance;
			    intersectionPoint = updateIntersectionPoint;
			    normal = updateNormal;
			    objectIndex = i;
			}
		  }
	  }

	  if(objectIndex != -1){
	    colors[index] = glm::vec3(0, 0, 0);
	    return;
	  }
	  
	  // Initialize the material index
	  int materialIndex = geoms[objectIndex].materialid;

	  // Determine if the object itself can be treated as light source
	  if(materials[materialIndex].emittance > 0){
		// If the object is a light source, then the color in the image is it own
		colors[index] = materials[materialIndex].color;
	  }else{
		// Find the inverse light ray from intersection point to a random point on the light source
		ray lightRay, inverseLightRay, reflectionRay;
		inverseLightRay.origin = intersectionPoint;
	    for(int i = 0; i < numberOfLights; ++ i){
		  int lightIndex = lights[i];
		  if(geoms[lightIndex].type == SPHERE){
		    lightRay.origin = getRandomPointOnSphere(geoms[lightIndex], time * index);
		  }else if(geoms[lightIndex].type == CUBE){
			lightRay.origin = getRandomPointOnCube(geoms[lightIndex], time * index);
		  }
		  inverseLightRay.direction = glm::normalize( (inverseLightRay.origin - lightRay.origin));
		  float lightDistance = glm::distance(inverseLightRay.origin,intersectionPoint);
		  // Using the distance from the light source to the intersection point to determine whether the light is obstructed by other objects
		  bool lightFlag = true;
		  for(int j = 0; j < numberOfGeoms; ++ j){
			if (j == lightIndex)
				continue;
		    if(geoms[j].type == SPHERE){
		      updateDistance = sphereIntersectionTest(geoms[j], inverseLightRay, updateIntersectionPoint, updateNormal);
			  if(updateDistance != -1 && updateDistance < lightDistance){
			    lightFlag = false;
			    break;
			  }
		    }else if (geoms[j].type == CUBE){
			  updateDistance = boxIntersectionTest(geoms[j], inverseLightRay, updateIntersectionPoint, updateNormal);
			  if(updateDistance != -1 && updateDistance < lightDistance){
				lightFlag = false;
				break;
			  }
		    }
			//
			if (lightFlag == true){
			  //Factors for diffusion and shading
			  float diffusionCoefficient = 0.7f;
			  float specularCoefficient  = 0.2f;
			  //Lambertian Surface Diffusion
			  float diffuse  = diffusionCoefficient * glm::max(glm::dot(inverseLightRay.direction, normal), 0.0f);
			  colors[index]  += diffuse * materials[geoms[lightIndex].materialid].color * materials[materialIndex].color;
			  //Phong Highlighting (Phong lighting equation: ambient + diffuse + specular =  phong reflection)
			  reflectionRay.direction = calculateReflectionDirection(normal, -inverseLightRay.direction);
			  float specular = specularCoefficient * pow(max((float)glm::dot(-cameraRay.direction, reflectionRay.direction), 0.0f), materials[materialIndex].specularExponent);
			  colors[index]  += specular * materials[geoms[lightIndex].materialid].color * materials[materialIndex].specularColor;
			}
		  }
	    }
	  }
    //colors[index] = generateRandomNumberFromThread(resolution, time, x, y);
   }
}

//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth = 1; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  // Initialize the light
  int numberOfLights = 0;
  int* lights =  new int[numberOfGeoms];

  //package geometry and materials and sent to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    geomList[i] = newStaticGeom;

	// Calculate the number of light and record the light source object index
	if( materials[newStaticGeom.materialid].emittance > 0 ){
		lights[numberOfLights] = i;
		++ numberOfLights;
	}
  }
  
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
  
  // Package materials
  glm::vec3* cudamaterials = NULL;
  cudaMalloc((void**)&cudamaterials, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudamaterials, materials, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);

  // Package lights
  int* cudalights = NULL;
  cudaMalloc((void**)&cudalights, numberOfLights * sizeof(lights));
  cudaMemcpy(cudalights, lights, numberOfLights * sizeof(lights), cudaMemcpyHostToDevice);

  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  //kernel launches
  raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, materials, numberOfMaterials, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, lights, numberOfLights);

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( cudamaterials );
  cudaFree( cudalights );
  delete geomList;
  delete lights;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}