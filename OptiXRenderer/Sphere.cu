#include <optix.h>
#include <optix_device.h>
#include "Geometries.h"

using namespace optix;

rtBuffer<Sphere> spheres; // a buffer of all spheres

rtDeclareVariable(Ray, ray, rtCurrentRay, );

// Attributes to be passed to material programs 
rtDeclareVariable(Attributes, attrib, attribute attrib, );

RT_PROGRAM void intersect(int primIndex)
{
	// Find the intersection of the current ray and sphere
	Sphere sphere = spheres[primIndex];
	float t;

	//	Transform Ray
	Ray invRay = ray;
	float4 origTrans = sphere.invTransform * make_float4( invRay.origin, 1 );
	invRay.origin = make_float3( origTrans / origTrans.w );
	invRay.direction = make_float3( sphere.invTransform * make_float4( invRay.direction, 0 ) );

	// sphere intersection test
	//	Solve quadratic eq
	float a = dot( invRay.direction, invRay.direction );
	float b = 2 * dot( invRay.direction, invRay.origin - sphere.center );
	float c = dot( invRay.origin - sphere.center, invRay.origin - sphere.center ) - (sphere.radius * sphere.radius);

	float t1, t2;

	float discriminant = b * b - 4 * a * c;

	if (discriminant < 0)
		return;
	else if (discriminant == 0)
	{
		t1 = t2 = -b / (2 * a);
	}
	else
	{
		//	Convert from normal Quadratic solution to a more stable form (to reduce loss of significance)
		float q;
		if (b >= 0)
		{
			q = -0.5f * (b + sqrt( discriminant ));
		}
		else
		{
			q = -0.5f * (b - sqrt( discriminant ));
		}
		t1 = q / a;
		t2 = c / q;

		//	Sort answer
		if (t1 > t2)
		{
			//	use q as a tmp var
			q = t1;
			t1 = t2;
			t2 = q;
		}
	}

	//	Check answer case
	if (t1 < 0)
	{
		//	if t1 is negative then use t2
		t1 = t2;

		if (t1 < 0)
		{
			return;
		}
	}

	t = t1;

	// Report intersection (material programs will handle the rest)
	if (rtPotentialIntersection(t))
	{
		//	Calculate the intersection point
		attrib.incomingRay = ray;
		attrib.incomingRay.tmin = t;
		attrib.incomingRay.tmax = t;

		//	Intersection data
		float3 invIntersectionPoint = invRay.origin + invRay.direction * t;
		float3 invNormal = invIntersectionPoint - sphere.center;

		attrib.normal = normalize( make_float3( sphere.transposeInvTransform * make_float4( invNormal, 0 ) ) );

		// Pass the material attributes
		attrib.shapeMaterial = sphere.shapeMaterial;

		rtReportIntersection(0);
	}
}

RT_PROGRAM void bound(int primIndex, float result[6])
{
	Sphere sphere = spheres[primIndex];

	//	Creating all BB corners of a sphere
	float4 bb_sphere[ 8 ];

	for (int x = 0; x <= 1; x++) 
	for (int y = 0; y <= 1; y++)
	for (int z = 0; z <= 1; z++)
		bb_sphere[ x * 4 + y * 2 + z ] = 
			make_float4( 
				( sphere.center - make_float3( sphere.radius ))
					+ make_float3( sphere.radius * 2 ) * make_float3( x, y, z )
				, 1 
			);

	float3 ll, ur;
	ll = make_float3( RT_DEFAULT_MAX );
	ur = make_float3( -RT_DEFAULT_MAX );

	//	Find minimum and maximum of each point
	for (int i = 0; i < 8; i++)
	{
		float4 vert = sphere.transform * bb_sphere[ i ];

		if (ll.x > vert.x)
			ll.x = vert.x;
		if (ll.y > vert.y)
			ll.y = vert.y;
		if (ll.z > vert.z)
			ll.z = vert.z;

		if (ur.x < vert.x)
			ur.x = vert.x;
		if (ur.y < vert.y)
			ur.y = vert.y;
		if (ur.z < vert.z)
			ur.z = vert.z;
	}


	result[ 0 ] = ll.x;
	result[ 1 ] = ll.y;
	result[ 2 ] = ll.z;
	result[ 3 ] = ur.x;
	result[ 4 ] = ur.y;
	result[ 5 ] = ur.z;

}