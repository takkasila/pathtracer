#include <optix.h>
#include <optix_device.h>

#include "Payloads.h"

using namespace optix;

rtDeclareVariable(Payload, payload, rtPayload, );
rtDeclareVariable(float3, backgroundColor, , );

// Attributes to be passed to material programs 
rtDeclareVariable( Attributes, attrib, attribute attrib, );

RT_PROGRAM void miss()
{
	// Set the result to be the background color if miss
	payload.radiance = backgroundColor;
	payload.done = true;
}

RT_PROGRAM void exception()
{
	// Print any exception for debugging
	const unsigned int code = rtGetExceptionCode();
	rtPrintExceptionDetails();
}

rtDeclareVariable(ShadowPayload, shadowPayload, rtPayload, );
rtDeclareVariable(float1, t, rtIntersectionDistance, );

RT_PROGRAM void shadowCaster_anyHit()
{
	//	If searching for anyLight
	if (shadowPayload.targetLightId == -1)
	{
		//	Check if this surface is a light source
		if (dot( attrib.shapeMaterial.emission, make_float3( 1 ) ) > 0)
		{
			shadowPayload.isVisible = true;
			shadowPayload.distToLight = t.x;
			shadowPayload.hitLightId = attrib.shapeMaterial.lightId;
			rtTerminateRay();
		}
	}
	//	Searching for a specific light/surface
	else
	{
		if ( t.x < shadowPayload.distToLight )
		{
			shadowPayload.isVisible = false;
			rtTerminateRay();
		}
	}
}

RT_PROGRAM void shadowCaster_closestHit()
{
	//	Check if it's a light source
	if (dot( attrib.shapeMaterial.emission, make_float3( 1 ) ) > 0)
	{
		//	If searching for any light
		if (shadowPayload.targetLightId == -1)
		{
			shadowPayload.isVisible = true;
			shadowPayload.distToLight = t.x;
			shadowPayload.hitLightId = attrib.shapeMaterial.lightId;
		}
		//	Or if searching for a specific light surface
		if (shadowPayload.targetLightId == attrib.shapeMaterial.lightId)
		{
			shadowPayload.isVisible = true;
			shadowPayload.distToLight = t.x;
			shadowPayload.hitLightId = attrib.shapeMaterial.lightId;
		}
	}
}