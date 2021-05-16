#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "random.h"

#include "Payloads.h"
#include "Geometries.h"
#include "Light.h"

using namespace optix;

#define	M_2PIf 2*M_PIf

// Declare variables
rtDeclareVariable(Payload, payload, rtPayload, );
rtDeclareVariable(rtObject, root, , );


// Declare light buffers
rtBuffer<PointLight> plights;
rtBuffer<DirectionalLight> dlights;
rtBuffer<QuadLight> qlights;

rtDeclareVariable( int, numPointLight, , );
rtDeclareVariable( int, numDirectionalLight, , );
rtDeclareVariable( int, numQuadLight, , );
rtDeclareVariable( int, numLightSample, , );
rtDeclareVariable( int, isStratify, , );
rtDeclareVariable( int, nexteventestimationMode, , );
rtDeclareVariable( int, importanceSamplingMode, , );

// Declare attibutes 
rtDeclareVariable(Attributes, attrib, attribute attrib, );

//
//	BRDF: Phong
//

//	Sample
static __host__ __device__ __inline__ float3 sampleBRDF_Phong( float3 omega_o )
{
	//	Sampling a point on a half hemishphere in polar coordinate for BRDF modified phong

	float meanDiffuse = (attrib.shapeMaterial.diffuse.x + attrib.shapeMaterial.diffuse.y + attrib.shapeMaterial.diffuse.z) / 3.0f;
	float meanSpecular = (attrib.shapeMaterial.specular.x + attrib.shapeMaterial.specular.y + attrib.shapeMaterial.specular.z) / 3.0f;

	float reflectiveness;
	if(meanDiffuse + meanSpecular == 0)
		reflectiveness = 1;
	else
		reflectiveness = meanSpecular / (meanDiffuse + meanSpecular);

	float samplingReflectivity = rnd( payload.prev_random );

	//	Reflection vector
	float3 r = reflect( attrib.incomingRay.direction, attrib.normal );

	//	The sampling coordinate frame
	float3 w = make_float3( 0 );

	float3 sampleInHermisphere = make_float3( 0 );

	if (samplingReflectivity <= reflectiveness)
	{
		//	Specular
		float theta = acosf( pow( rnd( payload.prev_random ), 1.0f / (attrib.shapeMaterial.shininess + 1.0f) ) );
		float phi = M_2PIf * rnd( payload.prev_random );

		sampleInHermisphere = make_float3(
			cos( phi ) * sin( theta )
			, sin( phi ) * sin( theta )
			, cos( theta )
		);

		//	To roate sampling coordinate frame to the reflection vector
		w = r;
	}
	else
	{
		//	Diffuse
		float theta = acosf( sqrt( rnd( payload.prev_random ) ) );
		float phi = M_2PIf * rnd( payload.prev_random );

		sampleInHermisphere = make_float3(
			cos( phi ) * sin( theta )
			, sin( phi ) * sin( theta )
			, cos( theta )
		);

		//	To roate sampling coordinate frame to the surface normal
		w = attrib.normal;
	}

	//
	//	Rotate the sampling point so that z-axis of sampling space match with the sampling coordinate frame
	//
	float3 up = normalize( make_float3( 0, 1, 0 ) );

	//	If the up vector is the same as centerOfSampling then change the up vector.
	//	Can check by their dot product
	if (length( cross( up, w ) ) == 0)
	{
		up = make_float3( 1, 0, 0 );
	}

	float3 u = normalize( cross( up, w ) );
	float3 v = cross( w, u );
	float3 omega_i = sampleInHermisphere.x * u + sampleInHermisphere.y * v + sampleInHermisphere.z * w;

	return omega_i;
}

//	Evaluate
static __host__ __device__ __inline__ float3 evaluateBRDF_Phong( float3 omega_o, float3 omega_i )
{
	//	Reflection vector
	float3 r = reflect( -omega_o, attrib.normal );

	float3 brdfTerm = (attrib.shapeMaterial.diffuse / M_PIf)
						+ attrib.shapeMaterial.specular * ((attrib.shapeMaterial.shininess + 2.0f) / M_2PIf)
														* pow( fmaxf( 0.0f, dot( r, omega_i ) ), attrib.shapeMaterial.shininess );

	return brdfTerm;
}

//	PDF
static __host__ __device__ __inline__ float calculatePDF_BRDF_Phong( float3 omega_o, float3 omega_i )
{
	float meanDiffuse = (attrib.shapeMaterial.diffuse.x + attrib.shapeMaterial.diffuse.y + attrib.shapeMaterial.diffuse.z) / 3.0f;
	float meanSpecular = (attrib.shapeMaterial.specular.x + attrib.shapeMaterial.specular.y + attrib.shapeMaterial.specular.z) / 3.0f;

	float reflectiveness;
	if(meanDiffuse + meanSpecular == 0)
		reflectiveness = 1;
	else
		reflectiveness = meanSpecular / (meanDiffuse + meanSpecular);

	//	Reflection vector
	float3 r = reflect( attrib.incomingRay.direction, attrib.normal );
	
	float pdfTerm = (1.0 - reflectiveness) * fmaxf( 0.0f, dot( attrib.normal, omega_i ) ) / M_PIf
					+ reflectiveness * ( ( attrib.shapeMaterial.shininess + 1.0f ) / M_2PIf)
										* pow( fmaxf( 0.0f, dot( r, omega_i ) ), attrib.shapeMaterial.shininess );
	return pdfTerm;
}

//
//	BRDF: GGX
//

//	Sample
static __host__ __device__ __inline__ float3 sampleBRDF_GGX( float3 omega_o )
{
	//	Sampling a point on a half hemishphere in polar coordinate for BRDF GGX

	float meanDiffuse = (attrib.shapeMaterial.diffuse.x + attrib.shapeMaterial.diffuse.y + attrib.shapeMaterial.diffuse.z) / 3.0f;
	float meanSpecular = (attrib.shapeMaterial.specular.x + attrib.shapeMaterial.specular.y + attrib.shapeMaterial.specular.z) / 3.0f;

	float reflectiveness;
	if (meanDiffuse + meanSpecular == 0)
		reflectiveness = 1;
	else
		reflectiveness = fmaxf( 0.25f, meanSpecular / (meanDiffuse + meanSpecular) );

	float samplingReflectivity = rnd( payload.prev_random );

	//	Reflection vector
	float3 r = reflect( attrib.incomingRay.direction, attrib.normal );

	//	The sampling coordinate frame
	float3 w = make_float3( 0 );

	float3 omega_i = make_float3( 0 );

	if (samplingReflectivity <= reflectiveness)
	{
		//	Specular
		//float theta = acosf( pow( rnd( payload.prev_random ), 1.0f / (attrib.shapeMaterial.shininess + 1.0f) ) );
		float xi = rnd( payload.prev_random );
		float thetaAtanPart = (attrib.shapeMaterial.roughness * sqrtf( xi )) / (sqrtf( 1 - xi ));
		float theta = atanf( thetaAtanPart );
		float phi = M_2PIf * rnd( payload.prev_random );

		float3 halfVector = make_float3(
			cos( phi ) * sin( theta )
			, sin( phi ) * sin( theta )
			, cos( theta )
		);

		//	To roate sampling coordinate frame to the reflection vector
		w = attrib.normal;

		//
		//	Rotate the sampling Half-Vector so that z-axis of sampling space match with the sampling coordinate frame
		//
		float3 up = normalize( make_float3( 0, 1, 0 ) );

		//	If the up vector is the same as centerOfSampling then change the up vector.
		//	Can check by their dot product
		if (length( cross( up, w ) ) == 0)
		{
			up = make_float3( 1, 0, 0 );
		}

		float3 u = normalize( cross( up, w ) );
		float3 v = cross( w, u );

		halfVector = halfVector.x * u + halfVector.y * v + halfVector.z * w;

		//	Calculate the sampling omega_i such that it reflected off Half-Vector from omega_o
		omega_i = reflect( -omega_o, halfVector );
	}
	else
	{
		//	Diffuse
		float theta = acosf( sqrt( rnd( payload.prev_random ) ) );
		float phi = M_2PIf * rnd( payload.prev_random );

		 float3 sampleInHermisphere = make_float3(
			cos( phi ) * sin( theta )
			, sin( phi ) * sin( theta )
			, cos( theta )
		);

		//	To roate sampling coordinate frame to the surface normal
		w = attrib.normal;

		//
		//	Rotate the sampling point so that z-axis of sampling space match with the sampling coordinate frame
		//
		float3 up = normalize( make_float3( 0, 1, 0 ) );

		//	If the up vector is the same as centerOfSampling then change the up vector.
		//	Can check by their dot product
		if (length( cross( up, w ) ) == 0)
		{
			up = make_float3( 1, 0, 0 );
		}

		float3 u = normalize( cross( up, w ) );
		float3 v = cross( w, u );
		omega_i = sampleInHermisphere.x * u + sampleInHermisphere.y * v + sampleInHermisphere.z * w;
	}

	return omega_i;
}

//	Evaluate
static __host__ __device__ __inline__ float3 evaluateBRDF_GGX( float3 omega_o, float3 omega_i, float &microDistFuncTerm )
{
	float3 halfVector = normalize( omega_o + omega_i );

	float divisorTerm = 4 * fmaxf( 0, dot( omega_i, attrib.normal ) ) * fmaxf( 0, dot( omega_o, attrib.normal ) );
	float3 brdfTerm_GGX = make_float3( 0 );
	microDistFuncTerm = 1;

	if (divisorTerm > 0)
	{
		float roughnessSquare = attrib.shapeMaterial.roughness * attrib.shapeMaterial.roughness;

		//	Fresnel term with Schlick's Approx
		float3 fresnelTerm = attrib.shapeMaterial.specular
			+ (make_float3( 1 ) - attrib.shapeMaterial.specular) * powf( 1.0f - fmaxf( 0, dot( omega_i, halfVector ) ), 5.0f );

		//	Microfacet shadow-masking function. We use Smith-G which is a product of out-going masking and incoming masking.
		//	Since we know that dot product of both out-going and incoming direction and above the surface ( dotProduct > 1 ),
		//	we don't have to repeatly check for this validity again
		float theta_out = acosf( dot( omega_o, attrib.normal ) );
		float maskingOutDir = 2.0f / (1 + sqrtf( 1 + roughnessSquare * powf( tanf( theta_out ), 2.0f ) ));

		float theta_in = acosf( dot( omega_i, attrib.normal ) );
		float maskingInDir = 2.0f / (1 + sqrtf( 1 + roughnessSquare * powf( tanf( theta_in ), 2.0f ) ));

		float maskingTermG = maskingOutDir * maskingInDir;

		//	Microfacet distribution function
		float theta_half = acosf( fmaxf( 0, dot( halfVector, attrib.normal ) ) );
		microDistFuncTerm = roughnessSquare / (M_PIf * powf( cosf( theta_half ), 4.0f )
			* powf(
				(roughnessSquare + powf( tanf( theta_half ), 2.0f ))
				, 2.0f
			));

		//	Finally we can calculate the BRDF term
		brdfTerm_GGX = fresnelTerm * maskingTermG * microDistFuncTerm / divisorTerm;
	}

	//	Calculate the complete BRDF term which is consists of Phong Diffuse BRDF and GGX Specular BRDF
	return attrib.shapeMaterial.diffuse / M_PIf + brdfTerm_GGX;
}

//	PDF
static __host__ __device__ __inline__ float calculatePDF_BRDF_GGX( float3 omega_o, float3 omega_i, float microDistFuncTerm )
{
	//	Calculate the PDF term

	float meanDiffuse = (attrib.shapeMaterial.diffuse.x + attrib.shapeMaterial.diffuse.y + attrib.shapeMaterial.diffuse.z) / 3.0f;
	float meanSpecular = (attrib.shapeMaterial.specular.x + attrib.shapeMaterial.specular.y + attrib.shapeMaterial.specular.z) / 3.0f;

	float reflectiveness;
		if (meanDiffuse + meanSpecular == 0)
			reflectiveness = 1;
		else
			reflectiveness = fmaxf( 0.25f, meanSpecular / (meanDiffuse + meanSpecular) );

	//	Calculate the half angle vector
	float3 halfVector = normalize( omega_o + omega_i );

	//	For some reason, when sampling a point on a reflective GGX surface, you will
	//	sample a halfVector instead and then reflect omega_o about that halfVector to get omega_i.
	//	A problem arrives when the sampled halfVector is almost perpendicular to the omega_o, the
	//	reflected vector omega_i will be almost parallel with the omega_o. 
	//	Thus, when we retriving a halfVector from omega_o and omega_i that almost parallel,
	//	you will get an unstable result that sometimes point back in the opposite direction.
	//	We can work around this case by checking if the retrived halfVector pointing inside
	//	into the surface then flipped it back.
	if (dot( halfVector, attrib.normal ) < 0)
		halfVector *= -1;

	float pdfTerm = (1.0f - reflectiveness) * fmaxf( 0.0f, dot( attrib.normal, omega_i ) ) / M_PIf
					+ reflectiveness * microDistFuncTerm * fmaxf( 0.0f, dot( attrib.normal, halfVector ) )
							/ (4.0f * fmaxf( 0.0f, dot( halfVector, omega_i ) ));

	return pdfTerm;
}

//
//	NEE: Direct Area Light
//

//	PDF
static __host__ __device__ float calculatePDF_NEE_AreaLight( float3 omega_i, int lightId )
{
	QuadLight quadLight;
	//	Search for the target quad light index
	for (int i = 0; i < numQuadLight; i++)
	{
		QuadLight currQuadLight = qlights[ i ];
		if (currQuadLight.lightId == lightId)
		{
			quadLight = currQuadLight;
			break;
		}
		else if (i == numQuadLight - 1)
		{
			rtPrintf( "ERROR NOT FOUND QUAD LIGHT AT TARGET INDEX" );
		}
	}

	//
	//	Test if omega_i intersect with this light
	//

	//	Calculate intersection point
	float epsilon = 0.001f;
	float3 intersectionPoint = attrib.incomingRay.origin + attrib.incomingRay.direction * (attrib.incomingRay.tmin);
	float3 shadowRayDir = omega_i;
	float3 shadowRayOrigin = intersectionPoint + shadowRayDir * epsilon;

	Ray shadowRay = make_Ray( shadowRayOrigin, shadowRayDir, 1, epsilon, RT_DEFAULT_MAX );
	ShadowPayload shadowPayload;
	shadowPayload.isVisible = false;
	shadowPayload.targetLightId = quadLight.lightId;

	//	Trace the shadow ray
	rtTrace( root, shadowRay, shadowPayload );

	float pdf;

	//	If the ray to light is not obstruct then calculate the PDF
	if (shadowPayload.isVisible)
	{
		float quadArea = length( cross( quadLight.ab, quadLight.ac ) );
		pdf = (shadowPayload.distToLight * shadowPayload.distToLight)
				/ (quadArea * dot( quadLight.normal, -omega_i ) );
	}
	else
	{
		pdf = 0;
	}


	return pdf;
}

// PDF 
static __host__ __device__ float calculatePDF_NEE_AreaLight_AllLights( float3 omega_i )
{
	float totalPDF = 0;
	for (int i = 0; i < numQuadLight; i++)
	{
		QuadLight quadLight = qlights[ i ];
		int lightId = quadLight.lightId;

		totalPDF += calculatePDF_NEE_AreaLight( omega_i, lightId );
	}

	totalPDF /= numQuadLight;

	return totalPDF;
}

RT_PROGRAM void closestHit_pathtracer()
{
	//	Calculate intersection point
	float epsilon = 0.001f;
	float3 intersectionPoint = attrib.incomingRay.origin + attrib.incomingRay.direction * (attrib.incomingRay.tmin);

	//
	//	Sampling incident direction according to each mode
	//

	float3 omega_o = -attrib.incomingRay.direction;
	float3 omega_i = make_float3( 0 );
	float3 incidentRadianceTerm = make_float3( 0 );

	//	If the ray is at first depth then add hit emissive surface
	if (nexteventestimationMode == 0 || ( nexteventestimationMode != 0 && payload.depth == 0 ) )
	{
		//	If the surface is emissive then add the emission value and terminate the ray
		//	NOT SUPPORT REFLECTIVE EMISSIVE SURFACE
		if (dot( attrib.shapeMaterial.emission, make_float3( 1 ) ) > 0)
		{
			//	Check if the ray is hit in front side of surface
			if (dot( omega_o, attrib.normal ) < 0)
			{
				payload.radiance += attrib.shapeMaterial.emission;
				payload.throughput = make_float3( 0 );
				payload.done = true;
				return;
			}
		}
	}

	if (importanceSamplingMode == 0)
	{
		//
		//	Uniform Hemisphere Importance Sampling
		//

		//	Sampling a point on a half hemishphere in polar coordinate uniformly
		float theta = acosf( rnd( payload.prev_random ) );
		float phi = M_2PIf * rnd( payload.prev_random );

		float3 sampleInHermisphere = make_float3(
			cos( phi ) * sin( theta )
			, sin( phi ) * sin( theta )
			, cos( theta )
		);

		//	Rotate the sampling point so that z-axis of sampling space match with the surface normal
		float3 w = attrib.normal;
		float3 up = normalize( make_float3( 0, 1, 0 ) );

		//	If the up vector is the same as w then change the up vector.
		//	Can check by their dot product
		if (length( cross( up, w ) ) == 0)
		{
			up = make_float3( 1, 0, 0 );
		}

		float3 u = normalize( cross( up, w ) );
		float3 v = cross( w, u );
		omega_i = sampleInHermisphere.x * u + sampleInHermisphere.y * v + sampleInHermisphere.z * w;

		//
		//	Calculate BRDF term
		//
		float3 brdfTerm = evaluateBRDF_Phong( omega_o, omega_i );

		//	Calculate the incident radiance term
		incidentRadianceTerm = M_2PIf * brdfTerm * fmaxf( 0.0f, dot( attrib.normal, omega_i ) );
	}
	else if (importanceSamplingMode == 1)
	{
		//
		//	Cosine Importance Sampling
		//

		//	Sampling a point on a half hemishphere in polar coordinate with cosine pdf
		float theta = acosf( sqrt( rnd( payload.prev_random ) ) );
		float phi = M_2PIf * rnd( payload.prev_random );

		float3 sampleInHermisphere = make_float3(
			cos( phi ) * sin( theta )
			, sin( phi ) * sin( theta )
			, cos( theta )
		);

		//	Rotate the sampling point so that z-axis of sampling space match with the surface normal
		float3 w = attrib.normal;
		float3 up = normalize( make_float3( 0, 1, 0 ) );

		//	If the up vector is the same as w then change the up vector.
		//	Can check by their dot product
		if (length( cross( up, w ) ) == 0)
		{
			up = make_float3( 1, 0, 0 );
		}

		float3 u = normalize( cross( up, w ) );
		float3 v = cross( w, u );
		omega_i = sampleInHermisphere.x * u + sampleInHermisphere.y * v + sampleInHermisphere.z * w;

		//
		//	Calculate BRDF term
		//
		float3 brdfTerm = evaluateBRDF_Phong( omega_o, omega_i );

		//	Calculate the incident radiance term
		incidentRadianceTerm = M_PIf * brdfTerm;
	}
	else if (importanceSamplingMode == 2)
	{
		//
		//	BRDF Importance Sampling
		//

		float meanDiffuse = (attrib.shapeMaterial.diffuse.x + attrib.shapeMaterial.diffuse.y + attrib.shapeMaterial.diffuse.z) / 3.0f;
		float meanSpecular = (attrib.shapeMaterial.specular.x + attrib.shapeMaterial.specular.y + attrib.shapeMaterial.specular.z) / 3.0f;

		if (attrib.shapeMaterial.brdfMode == 0)
		{
			//
			//	Phong BRDF
			//
			
			// 	Sampling incdent direction
			omega_i = sampleBRDF_Phong( omega_o );

			//	Calculate BRDF term
			float3 brdfTerm = evaluateBRDF_Phong( omega_o, omega_i );

			//	Calculate the PDF term
			float pdfTerm = calculatePDF_BRDF_Phong( omega_o, omega_i );

			//	Calculate the incident radiance term
			incidentRadianceTerm = brdfTerm * fmaxf( 0.0f, dot( attrib.normal, omega_i ) ) / pdfTerm;
		}
		else if (attrib.shapeMaterial.brdfMode == 1)
		{
			//
			//	GGX BRDF
			//

			//	Sampling a point on a half hemishphere in polar coordinate for BRDF GGX
			omega_i = sampleBRDF_GGX( omega_o );

			//	Calculate the GGX BRDF term
			float microDistFuncTerm;
			float3 brdfTerm = evaluateBRDF_GGX( omega_o, omega_i, microDistFuncTerm );

			//	Calculate the PDF term
			float pdfTerm = calculatePDF_BRDF_GGX( omega_o, omega_i, microDistFuncTerm );

			//	Calculate the incident radiance term
			incidentRadianceTerm = brdfTerm * fmaxf( 0.0f, dot( attrib.normal, omega_i ) ) / pdfTerm;
		}
	}

	//	
	//	If nextEventEstimation mode is enabled then sample direct light
	//
	float3 directLightSampleAccumulate = make_float3( 0 );
	if (nexteventestimationMode == 1)
	{
		//	Check if the intersected surface is emissive
		//	If it is and the current ray depth is not 0 (first depth) then terminate
		//	the ray without adding any contribution at all 
		if (payload.depth > 0 && dot( attrib.shapeMaterial.emission, make_float3( 1 ) ) > 0)
		{
			payload.done = true;
			payload.radiance = make_float3( 0 );
			payload.throughput = make_float3( 0 );
			return;
		}

		//
		//	Sample Quad Light 
		//		For now sample every QuadLight with one ray per quad
		//	
		for (int i = 0; i < numQuadLight; i++)
		{
			QuadLight quadLight = qlights[ i ];
			float quadArea = length( cross( quadLight.ab, quadLight.ac ) );

			float3 lightSamplingPoint = quadLight.a + rnd( payload.prev_random ) * quadLight.ab + rnd( payload.prev_random ) * quadLight.ac;

			//	Light dir
			float3 omega_i = normalize( lightSamplingPoint - intersectionPoint );
			//	Out dir
			float3 omega_o = -attrib.incomingRay.direction;

			//
			//	Calculate Visibility term
			//
			float3 shadowRayDir = omega_i;
			float3 shadowRayOrigin = intersectionPoint + shadowRayDir * epsilon;

			Ray shadowRay = make_Ray( shadowRayOrigin, shadowRayDir, 1, epsilon, RT_DEFAULT_MAX );
			ShadowPayload shadowPayload;
			shadowPayload.isVisible = false;
			shadowPayload.targetLightId = quadLight.lightId;

			//	Trace the shadow ray
			rtTrace( root, shadowRay, shadowPayload );

			//	If the ray to light obstruct, then overall value is 0 so we can continue to the next sample point
			if (!shadowPayload.isVisible)
			{
				continue;
			}

			//
			//	Calculate BRDF term
			//
			float3 brdfTerm = make_float3( 0 );
			if (attrib.shapeMaterial.brdfMode == 0)
			{
				//	Phong BRDF
				brdfTerm = evaluateBRDF_Phong( omega_o, omega_i );
			}
			else if (attrib.shapeMaterial.brdfMode == 1)
			{
				//	GGX BRDF
				float microDistFuncTerm;
				brdfTerm = evaluateBRDF_GGX( omega_o, omega_i, microDistFuncTerm );
			}

			//
			//	Calculate Geometric term
			//
			float geometricTerm = 1.0f / (shadowPayload.distToLight * shadowPayload.distToLight)
				* fmaxf( 0.0f, dot( attrib.normal, omega_i ) )
				* fmaxf( 0.0f, dot( quadLight.normal, -omega_i ) );

			directLightSampleAccumulate += quadLight.intensity * (quadArea / numLightSample) * brdfTerm * geometricTerm;
		}
	}
	else if (nexteventestimationMode == 2)
	{
		//
		//	Multiple Importance Sampling light source:
		//		- NEE Area Light Sampling
		//		- BRDF Sampling
		//

		//	Check if the intersected surface is emissive
		//	If it is and the current ray depth is not 0 (first depth) then terminate
		//	the ray without adding any contribution at all 
		if (payload.depth > 0 && dot( attrib.shapeMaterial.emission, make_float3( 1 ) ) > 0)
		{
			payload.done = true;
			payload.radiance = make_float3( 0 );
			payload.throughput = make_float3( 0 );
			return;
		}

		//
		//	Calculate NEE Sampling
		//

		//
		//	NEE Area Light
		//
		float3 estimatedNEE_AreaLight = make_float3( 0, 0, 0 );
		for (int i = 0; i < numQuadLight; i++)
		{
			QuadLight quadLight = qlights[ i ];
			float quadArea = length( cross( quadLight.ab, quadLight.ac ) );

			float3 lightSamplingPoint = quadLight.a + rnd( payload.prev_random ) * quadLight.ab + rnd( payload.prev_random ) * quadLight.ac;

			//	Light dir
			float3 omega_i = normalize( lightSamplingPoint - intersectionPoint );
			//	Out dir
			float3 omega_o = -attrib.incomingRay.direction;

			//
			//	Calculate Visibility term
			//
			float3 shadowRayDir = omega_i;
			float3 shadowRayOrigin = intersectionPoint + shadowRayDir * epsilon;

			Ray shadowRay = make_Ray( shadowRayOrigin, shadowRayDir, 1, epsilon, RT_DEFAULT_MAX );
			ShadowPayload shadowPayload;
			shadowPayload.isVisible = false;
			shadowPayload.targetLightId = quadLight.lightId;

			//	Trace the shadow ray
			rtTrace( root, shadowRay, shadowPayload );

			//	If the ray to light obstruct, then overall value is 0 so we can continue to the next sample point
			if (!shadowPayload.isVisible)
			{
				continue;
			}

			//
			//	Calculate BRDF term
			//
			float3 brdfTerm = make_float3( 0 );
			if (attrib.shapeMaterial.brdfMode == 0)
			{
				//	Phong BRDF
				brdfTerm = evaluateBRDF_Phong( omega_o, omega_i );
			}
			else if (attrib.shapeMaterial.brdfMode == 1)
			{
				//	GGX BRDF
				float microDistFuncTerm;
				brdfTerm = evaluateBRDF_GGX( omega_o, omega_i, microDistFuncTerm );
			}

			//
			//	Calculate Geometric term
			//
			float geometricTerm = 1.0f / (shadowPayload.distToLight * shadowPayload.distToLight)
				* fmaxf( 0.0f, dot( attrib.normal, omega_i ) )
				* fmaxf( 0.0f, dot( quadLight.normal, -omega_i ) );

			float3 evaluatedAreaLight = quadLight.intensity * (quadArea / numLightSample) * brdfTerm * geometricTerm;

			//
			//	PDF 
			//
			float pdf_NEE_AreaLight = calculatePDF_NEE_AreaLight_AllLights( omega_i );

			//
			//	Weight
			//
			float pdf_NEE_BRDF = 0;
			if (attrib.shapeMaterial.brdfMode == 0)
			{
				pdf_NEE_BRDF = calculatePDF_BRDF_Phong( omega_o, omega_i );
			}
			else if (attrib.shapeMaterial.brdfMode == 1)
			{
				float microDistFuncTerm;
				pdf_NEE_BRDF = calculatePDF_BRDF_GGX( omega_o, omega_i, microDistFuncTerm );
			}

			//	Using Exponent = 2
			float weightNEE = ( pdf_NEE_AreaLight * pdf_NEE_AreaLight )  
								/ (pdf_NEE_AreaLight * pdf_NEE_AreaLight + pdf_NEE_BRDF * pdf_NEE_BRDF );


			//	TODO: Fix this

			//	Finally calculated the estimated NEE
			//estimatedNEE_AreaLight += weightNEE * evaluatedAreaLight / pdf_NEE_AreaLight;

			estimatedNEE_AreaLight += evaluatedAreaLight;
		}
		//estimatedNEE_AreaLight /= numQuadLight;


		//
		//	NEE BRDF
		//
		float3 estimatedNEE_BRDF = make_float3( 0, 0, 0 );

		//	Light dir
		float3 omega_i;
		if (attrib.shapeMaterial.brdfMode == 0)
		{
			omega_i = sampleBRDF_Phong( omega_o );
		}
		else if (attrib.shapeMaterial.brdfMode == 1)
		{
			omega_i = sampleBRDF_GGX( omega_o );
		}

		//	Out dir
		float3 omega_o = -attrib.incomingRay.direction;


		//
		//	Calculate Visibility term
		//
		float3 shadowRayDir = omega_i;
		float3 shadowRayOrigin = intersectionPoint + shadowRayDir * epsilon;

		Ray shadowRay = make_Ray( shadowRayOrigin, shadowRayDir, 1, epsilon, RT_DEFAULT_MAX );
		ShadowPayload shadowPayload;
		shadowPayload.isVisible = false;
		//	Target any light
		shadowPayload.targetLightId = -1;

		//	Trace the shadow ray
		rtTrace( root, shadowRay, shadowPayload );

		//	If hit a light
		if (shadowPayload.isVisible)
		{
			//	Search for the quad with that lightId
			int hitQuadIndex = 0;
			for( ; hitQuadIndex < numQuadLight; hitQuadIndex++ )
			{
				QuadLight currQuadLight = qlights[ hitQuadIndex ];
				//	If found
				if (currQuadLight.lightId == shadowPayload.hitLightId)
				{
					break;
				}
				//	If not found
				else if (hitQuadIndex == numQuadLight - 1)
				{
					rtPrintf( "ERROR! FIND QUADLIGHT AT INDEX\n" );
				}
			}

			QuadLight quadLight = qlights[ hitQuadIndex ];

			//
			//	Calculate BRDF term
			//
			float3 brdfTerm = make_float3( 0 );
			if (attrib.shapeMaterial.brdfMode == 0)
			{
				//	Phong BRDF
				brdfTerm = evaluateBRDF_Phong( omega_o, omega_i );
			}
			else if (attrib.shapeMaterial.brdfMode == 1)
			{
				//	GGX BRDF
				float microDistFuncTerm;
				brdfTerm = evaluateBRDF_GGX( omega_o, omega_i, microDistFuncTerm );
			}

			//
			//	Calculate Geometric term
			//
			float geometricTerm = 1.0f / (shadowPayload.distToLight * shadowPayload.distToLight)
				* fmaxf( 0.0f, dot( attrib.normal, omega_i ) )
				* fmaxf( 0.0f, dot( quadLight.normal, -omega_i ) );

			float quadArea = length( cross( quadLight.ab, quadLight.ac ) );

			float3 evaluatedBRDFLight = quadLight.intensity * (quadArea / numLightSample) * brdfTerm * geometricTerm;

			//
			//	PDF 
			//
			float pdf_NEE_BRDF = 0;
			if (attrib.shapeMaterial.brdfMode == 0)
			{
				pdf_NEE_BRDF = calculatePDF_BRDF_Phong( omega_o, omega_i );
			}
			else if (attrib.shapeMaterial.brdfMode == 1)
			{
				float microDistFuncTerm;
				pdf_NEE_BRDF = calculatePDF_BRDF_GGX( omega_o, omega_i, microDistFuncTerm );
			}

			//
			//	Weight
			//
			float pdf_NEE_AreaLight = calculatePDF_NEE_AreaLight_AllLights( omega_i );
			//float pdf_NEE_AreaLight = calculatePDF_NEE_AreaLight( omega_i, quadLight.lightId );

			//	Using Exponent = 2
			float weightNEE = (pdf_NEE_BRDF * pdf_NEE_BRDF)
				/ (pdf_NEE_BRDF * pdf_NEE_BRDF + pdf_NEE_AreaLight * pdf_NEE_AreaLight);


			//	Finally calculated the estimated NEE
			//estimatedNEE_BRDF += weightNEE * evaluatedBRDFLight / pdf_NEE_BRDF;
			estimatedNEE_BRDF += evaluatedBRDFLight;
		}
		//	If not then don't do anything
		else
		{
			estimatedNEE_BRDF = make_float3( 0, 0, 0 );
		}

		directLightSampleAccumulate = estimatedNEE_AreaLight + estimatedNEE_BRDF;
	}

	//	Pass data to Payload
	payload.radiance = attrib.shapeMaterial.ambient + directLightSampleAccumulate;
	
	payload.throughput *= incidentRadianceTerm;

	payload.dir = omega_i;

	payload.origin = intersectionPoint + payload.dir * epsilon;

}