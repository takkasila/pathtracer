/* 
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *	notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *	notice, this list of conditions and the following disclaimer in the
 *	documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *	contributors may be used to endorse or promote products derived
 *	from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <optixu/optixu_math_namespace.h>

template<unsigned int N>
static __host__ __device__ __inline__ unsigned int tea( unsigned int val0, unsigned int val1 )
{
  unsigned int v0 = val0;
  unsigned int v1 = val1;
  unsigned int s0 = 0;

  for( unsigned int n = 0; n < N; n++ )
  {
	s0 += 0x9e3779b9;
	v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
	v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
  }

  return v0;
}

// Generate random unsigned int in [0, 2^24)
static __host__ __device__ __inline__ unsigned int lcg(unsigned int &prev)
{
  const unsigned int LCG_A = 1664525u;
  const unsigned int LCG_C = 1013904223u;
  prev = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

static __host__ __device__ __inline__ unsigned int lcg2(unsigned int &prev)
{
  prev = (prev*8121 + 28411)  % 134456;
  return prev;
}

// Generate random float in [0, 1)
static __host__ __device__ __inline__ float rnd(unsigned int &prev)
{
  return ((float) lcg(prev) / (float) 0x01000000);
}

static __host__ __device__ __inline__ unsigned int rot_seed( unsigned int seed, unsigned int frame )
{
	return seed ^ frame;
}

/*
	The below noise functions are from
	https://gist.github.com/patriciogonzalezvivo/670c22f3966e662d2f83
*/
//	3D Noise function
static __host__ __device__ __inline__ float mod289( float x ){ return x - floor( x * (1.0 / 289.0) ) * 289.0; }
static __host__ __device__ __inline__ float4 mod289( float4 x ){ return x - floor( x * (1.0 / 289.0) ) * 289.0; }
static __host__ __device__ __inline__ float4 perm( float4 x ){ return mod289( ((x * 34.0) + 1.0) * x ); }

static __host__ __device__ __inline__ float noise( float3 p ){
	float3 a = floor( p );
	float3 d = p - a;
	d = d * d * (3.0 - 2.0 * d);

	float4 b = make_float4( a.x, a.x, a.y, a.y ) + make_float4( 0.0, 1.0, 0.0, 1.0 );
	float4 k1 = perm( make_float4( b.x, b.y, b.x, b.y ) );
	float4 k2 = perm( make_float4( k1.x, k1.y, k1.x, k1.y ) + make_float4( b.z, b.z, b.w, b.w ) );

	float4 c = k2 + make_float4( a.z );
	float4 k3 = perm( c );
	float4 k4 = perm( c + 1.0 );

	//float4 o1 = fract( k3 * (1.0 / 41.0) );
	float real;
	//float4 o1 = modff( k3.x * (1.0 / 41.0), &real );
	float4 o1 = make_float4(
		modff( k3.x * (1.0 / 41.0), &real )
		, modff( k3.y * (1.0 / 41.0), &real )
		, modff( k3.z * (1.0 / 41.0), &real )
		, modff( k3.w * (1.0 / 41.0), &real )
	);
	//float4 o2 = fract( k4 * (1.0 / 41.0) );
	float4 o2 = make_float4(
		modff( k4.x * (1.0 / 41.0), &real )
		, modff( k4.x * (1.0 / 41.0), &real )
		, modff( k4.x * (1.0 / 41.0), &real )
		, modff( k4.x * (1.0 / 41.0), &real )
	);

	float4 o3 = o2 * d.z + o1 * (1.0 - d.z);
	float2 o4 = make_float2( o3.y, o3.w ) * d.x + make_float2( o3.x, o3.z ) * (1.0 - d.x);

	return o4.y * d.y + o4.x * (1.0 - d.y);
}
