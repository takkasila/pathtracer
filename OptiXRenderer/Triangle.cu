#include "optix.h"
#include "optix_device.h"
#include "Geometries.h"

using namespace optix;

rtBuffer<Triangle> triangles; // a buffer of all spheres

rtDeclareVariable(Ray, ray, rtCurrentRay, );

// Attributes to be passed to material programs 
rtDeclareVariable(Attributes, attrib, attribute attrib, );

RT_PROGRAM void intersect(int primIndex)
{
	// Find the intersection of the current ray and triangle
	Triangle tri = triangles[primIndex];

	//	Transform Ray
	Ray invRay = ray;
	float4 origTrans = tri.invTransform * make_float4( invRay.origin, 1 );
	invRay.origin = make_float3( origTrans / origTrans.w );
	invRay.direction = make_float3( tri.invTransform * make_float4( invRay.direction, 0 ) );

	float t, u, v;

	//	Ray-plane intersection
	float3 triangleNormal = cross( tri.v2 - tri.v1, tri.v3 - tri.v1 );
	float triangleArea = length( triangleNormal ) / 2.0f;
	triangleNormal = normalize( triangleNormal );

	t = ( dot( tri.v1, triangleNormal ) - dot( invRay.origin, triangleNormal ) ) 
		/ (dot( invRay.direction, triangleNormal ));

	//	Check if triangle is behind
	if (t < 0)
		return;

	//	Check if Ray and Triangle is parallel
	if (dot( triangleNormal, invRay.direction ) == 0)
		return;

	//	Calculate intersection point
	float3 intersectionPoint = invRay.origin + invRay.direction * t;

	//	Check if hit point is inside triangle with Bary-Centric

	//	Edge 1
	float3 edge1 = tri.v2 - tri.v1;
	float3 tri1ToPoint = intersectionPoint - tri.v1;
	float3 crossProduct = cross( edge1, tri1ToPoint );
	//	If the crossProduct vector is in the opposite direction of the triangle's normal
	//		then the intersection point is outside of the triangle
	if (dot( triangleNormal, crossProduct ) < 0)
		return;

	//	Edge 2
	//	The same calculation goes for Edge 2 and Edge 3, except that we now also
	//		calculate for 'u' and 'v' barycentric coordinate since we substitute w = 1 - u - v
	float3 edge2 = tri.v3 - tri.v2;
	float3 tri2ToPoint = intersectionPoint - tri.v2;
	crossProduct = cross( edge2, tri2ToPoint );
	//u = ( length( crossProduct ) / 2.0f ) / triangleArea; ???
	u = dot( triangleNormal, crossProduct );
	if (u < 0)
		return;

	//	Edge 3
	float3 edge3 = tri.v1 - tri.v3;
	float3 tri3ToPoint = intersectionPoint - tri.v3;
	crossProduct = cross( edge3, tri3ToPoint );
	//v = (length( crossProduct ) / 2.0f) / triangleArea;
	v = dot( triangleNormal, crossProduct );
	if (v < 0)
		return;
	
	// Report intersection (material programs will handle the rest)
	if (rtPotentialIntersection(t))
	{
		//	Calculate the intersection point
		attrib.incomingRay = ray;
		attrib.incomingRay.tmin = t;
		attrib.incomingRay.tmax = t;

		//	Intersection data
		attrib.normal = normalize( make_float3( tri.transposeInvTransform * make_float4( triangleNormal, 0 ) ) );

		// Pass the material attributes
		attrib.shapeMaterial = tri.shapeMaterial;


		rtReportIntersection(0);
	}
}

RT_PROGRAM void bound(int primIndex, float result[6])
{
	Triangle tri = triangles[primIndex];

	//	Transform all points
	float3 ll, ur;
	ll = make_float3( RT_DEFAULT_MAX );
	ur = make_float3( -RT_DEFAULT_MAX );
	float3 triVerts[ 3 ] = { tri.v1, tri.v2, tri.v3 };

	//	Find minimum and maximum of each point
	for (int i = 0; i < 3; i++)
	{
		float3 vert = make_float3(  tri.transform * make_float4( triVerts[ i ], 1 ) );

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