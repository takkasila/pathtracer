#pragma once

#include <optixu/optixu_math_namespace.h>
#include "Geometries.h"

/**
 * Structures describing different payloads should be defined here.
 */

struct Payload
{
	optix::float3 origin;
	optix::float3 dir;
	optix::float3 radiance;
	bool done;
	int depth;
	unsigned int prev_random;
	optix::float3 throughput;
	bool isDebug;
};

struct ShadowPayload
{
	bool isVisible;
	int targetLightId;
	float distToLight;
	int hitLightId;
};