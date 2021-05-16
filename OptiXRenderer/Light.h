#pragma once

#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

/**
 * Structures describing different light sources should be defined here.
 */

struct PointLight
{
	optix::float3 position;
	optix::float3 color;
	optix::float3 attenuation;
};

struct DirectionalLight
{
	optix::float3 direction;
	optix::float3 color;
};

struct QuadLight
{
	optix::float3 a;
	optix::float3 ab;
	optix::float3 ac;
	optix::float3 intensity;
	optix::float3 normal;
	//	LightID
	//	0: Not a light
	//	1-N: Unique Light ID
	int lightId;
};