#pragma once

#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

#include "Geometries.h"
#include "Light.h"

struct Scene
{
	// Info about the output image
	std::string outputFilename;
	unsigned int width, height;
	float gamma;

	std::string integratorName;

	int maxdepth;
	
	//	for light source sampling
	int lightsamples;
	bool lightstratify;
	int nexteventestimationMode;

	//	path tracer
	int samplePerPixel;
	bool russianroulette;
	int importanceSamplingMode;

	std::vector<optix::float3> vertices;

	std::vector<Triangle> triangles;
	std::vector<Sphere> spheres;

	std::vector<DirectionalLight> dlights;
	std::vector<PointLight> plights;
	std::vector<QuadLight> qLights;

	//  Cameras
	optix::float3 camPos, camLookAt, camUp;
	float camFov;


	Scene()
	{
		//	Default Values
		this->outputFilename = "pathtrace.png";
		this->integratorName = "pathtracer";
		this->gamma = 1;
		this->maxdepth = 5;
		this->lightsamples = 1;
		this->lightstratify = false;
		this->samplePerPixel = 1;
		this->nexteventestimationMode = 0;
		this->russianroulette = false;
		this->importanceSamplingMode = 0;
	}
};