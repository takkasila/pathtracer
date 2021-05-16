#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "random.h"

#include "Payloads.h"

using namespace optix;

rtBuffer<float3, 2> resultBuffer; // used to store the render result

rtDeclareVariable(rtObject, root, , ); // Optix graph

rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, ); // a 2d index (x, y)

rtDeclareVariable(int1, frameID, , );

//	Scene Info
rtDeclareVariable( uint2, imgSize, , );
rtDeclareVariable( int, maxdepth, , );
rtDeclareVariable( int, samplePerPixel, , );
rtDeclareVariable( int, russianroulette, , );
rtDeclareVariable( float, gamma, , );

//	Camera Info 
rtDeclareVariable( float3, camPos, , );
rtDeclareVariable( float3, camLookAt, , );
rtDeclareVariable( float3, camUp, , );
rtDeclareVariable( float1, camFov, , );

//#define DEBUG_PIXEL
#ifdef DEBUG_PIXEL
//	X from Left to Right
//	Y from Bottom to Top

//	Cornell Light Center: 240, 430

//	GGX Test Green Ball: 578, 80

//	MIS Test, Floor : 600, 27
//	MIS Test, BrightLight : 600, 282

#define COORD_X 600
#define COORD_Y 27

#define isDebug( x, y ) (x == COORD_X && y == COORD_Y)

#endif // DEBUG_PIXEL



RT_PROGRAM void generateRays()
{

	//	epsilon
	float epsilon = 0.001f;

	// Calculate the ray direction (change the following lines)
	float3 u, v, w;

	w = normalize( camPos - camLookAt );
	u = normalize( cross( camUp, w ) );
	v = normalize( cross( w, u ) );

	//	assume that the given fov is fov-Y
	float tanFovY = tanf( ( camFov.x * ( M_PIf / 180.f ) ) / 2.0f );
	float tanFovX = tanFovY * (imgSize.x / float( imgSize.y ));

	//
	//	Sampling per pixel
	//

	//	Setting Up Random Number Generator
	unsigned int seed = 2 ^ 100000 ^ ( launchIndex.x * ( launchIndex.y ^ imgSize.x ) ) ;
	unsigned int prev_random = rot_seed( seed, frameID.x );

	float3 accumulateSPP = make_float3( 0 );
	for (int sppCount = 0; sppCount < samplePerPixel; sppCount++)
	{
#ifdef DEBUG_PIXEL
		if ( isDebug( launchIndex.x, launchIndex.y ) )
			rtPrintf( "\n\n=====v===New Sample!=====v=====\n" );
#endif // DEBUG_PIXEL

		float2 posInPixel = make_float2( 0 );

		//	Maintain the backward compatibility that if SPP is 1 then shoot the ray at the middle of the pixel.
		//	Otherwise random it.
		if (sppCount == 0)
			posInPixel = make_float2( 0.5, 0.5 );
		else
			posInPixel = make_float2( rnd( prev_random ) , rnd( prev_random ) );

		//	Calculate delta in X and Y. 
		//		Note that we shift the launchIndex by 0.5 to make it be at the center of pixel.
		float2 delta;
		delta.x = tanFovX * ((float( launchIndex.x + posInPixel.x ) - (imgSize.x / 2.0f)) / (imgSize.x / 2.0f));
		delta.y = tanFovY * (((imgSize.y / 2.0f) - float( launchIndex.y + posInPixel.y )) / (imgSize.y / 2.0f));

		//	invert-Y
		delta.y *= -1;

		//	Calculate Ray origin and direction
		float3 rayOrigin = camPos;
		float3 rayDir;
		rayDir = normalize( -w + delta.x * u + delta.y * v );


		// Shoot a ray to compute the color of the current pixel
		//	And also recursively traced it till hit the maximum depth
		Payload payload;
		payload.origin = rayOrigin;
		payload.dir = rayDir;
		payload.depth = 0;
		payload.done = false;
		payload.prev_random = prev_random;
		payload.throughput = make_float3( 1 );
		payload.isDebug = false;
#ifdef DEBUG_PIXEL
		if ( isDebug( launchIndex.x, launchIndex.y ) )
		{
			payload.isDebug = true;
			rtPrintf( "\npayload origin: %.4f %.4f %.4f\n", payload.origin.x, payload.origin.y, payload.origin.z );
			rtPrintf( "payload dir: %.4f %.4f %.4f\n", payload.dir.x, payload.dir.y, payload.dir.z );
		}
#endif

		float3 accumulatePerRay = make_float3( 0 );

		float3 prev_throughput = payload.throughput;

		while (!payload.done && ( russianroulette? true : payload.depth < maxdepth ) )
		{
			Ray ray = make_Ray( payload.origin, payload.dir, 0, epsilon, RT_DEFAULT_MAX );
			rtTrace( root, ray, payload );

			//	Accumulate the current radiance
			float3 currentRadiance = payload.radiance * prev_throughput;
			accumulatePerRay += currentRadiance;
			//accumulatePerRay = payload.radiance;
#ifdef DEBUG_PIXEL
			if ( isDebug( launchIndex.x, launchIndex.y ) )
			{
				rtPrintf( "payload radiance: %.4f %.4f %.4f\n", payload.radiance.x, payload.radiance.y, payload.radiance.z );
				rtPrintf( "payload throughput: %.4f %.4f %.4f\n", payload.throughput.x, payload.throughput.y, payload.throughput.z );
				rtPrintf( "prev_throughput: %.4f %.4f %.4f\n", prev_throughput.x, prev_throughput.y, prev_throughput.z );
				rtPrintf( "currentRadiance: %.4f %.4f %.4f\n", currentRadiance.x, currentRadiance.y, currentRadiance.z );
				rtPrintf( "accumulate: %.4f %.4f %.4f\n", accumulatePerRay.x, accumulatePerRay.y, accumulatePerRay.z );
				rtPrintf( "payload origin: %.4f %.4f %.4f\n", payload.origin.x, payload.origin.y, payload.origin.z );
				rtPrintf( "payload dir: %.4f %.4f %.4f\n\n", payload.dir.x, payload.dir.y, payload.dir.z );
			}
#endif // DEBUG_PIXEL

			prev_throughput = payload.throughput;

			//	If the russian roulette termination is enable
			if (russianroulette)
			{
				float terminationProbability = 1.0f - min( max( payload.throughput.x, max( payload.throughput.y, payload.throughput.z ) ), 1.0f );

				float prob = rnd( payload.prev_random );

				//	Terminate the ray
				if (prob < terminationProbability)
				{
					payload.done = true;
				}
				//	Boost the next output radiance
				else
				{
					prev_throughput /= (1.0f - terminationProbability);
				}

				payload.throughput = prev_throughput;
			}

			//	Update parameter for next recursive call
			//	Assume that the update of Ray origin and Direction is handled by the trace function
			payload.depth += 1;
		}

#ifdef DEBUG_PIXEL
		if ( isDebug( launchIndex.x, launchIndex.y ) )
		{
			rtPrintf( "payload terminate\n" );
		}
#endif // DEBUG_PIXEL

		accumulateSPP += accumulatePerRay;

		//	Update random seed
		prev_random = payload.prev_random;
	}

	accumulateSPP /= float( samplePerPixel );

	//	Gamma correction
	float oneOverGamma = 1.0f / gamma;
	accumulateSPP.x = pow( accumulateSPP.x, oneOverGamma );
	accumulateSPP.y = pow( accumulateSPP.y, oneOverGamma );
	accumulateSPP.z = pow( accumulateSPP.z, oneOverGamma );

	// Write the result
	resultBuffer[launchIndex] = accumulateSPP;

#ifdef DEBUG_PIXEL
	if ( isDebug( launchIndex.x, launchIndex.y ) )
	{
		rtPrintf( "final radiance: %.4f %.4f %.4f\n", accumulateSPP.x, accumulateSPP.y, accumulateSPP.z );
	}
#endif // DEBUG_PIXEL
}