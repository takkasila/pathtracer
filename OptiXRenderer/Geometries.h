#pragma once

#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

/**
 * Structures describing different geometries should be defined here.
 */

struct ShapeMaterial
{
	int brdfMode;
	optix::float3 ambient;
	optix::float3 emission;
	optix::float3 diffuse;
	optix::float3 specular;
	float shininess;
	float roughness;
	//	LightID
	//	0: Not a light
	//	1-N: Unique Light ID
	int lightId;
};

//	Attributes to pass to other RT_PROGRAM
struct Attributes
{
	ShapeMaterial shapeMaterial;

	optix::Ray incomingRay;

	optix::float3 normal;
};

struct Triangle
{
public:
	optix::float3 v1, v2, v3;

	optix::Matrix4x4 transform;
	optix::Matrix4x4 invTransform;
	optix::Matrix4x4 transposeInvTransform;

	ShapeMaterial shapeMaterial;

public:
	Triangle( optix::float3 v1, optix::float3 v2, optix::float3 v3 )
		:	v1( v1 ), v2( v2 ), v3( v3 )
	{}
};

struct Sphere
{
public:
	optix::float3 center;
	float radius;

	optix::Matrix4x4 transform;
	optix::Matrix4x4 invTransform;
	optix::Matrix4x4 transposeInvTransform;

	ShapeMaterial shapeMaterial;

public:
	Sphere( optix::float3 center, float radius )
		:	center( center ), radius( radius )
	{}
};
