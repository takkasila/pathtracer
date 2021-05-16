#include "SceneLoader.h"
#include <vector>

void SceneLoader::rightMultiply(const optix::Matrix4x4& M)
{
	optix::Matrix4x4& T = this->transStack.top();
	T = T * M;
}

optix::float3 SceneLoader::transformPoint(optix::float3 v)
{
	optix::float4 vh = this->transStack.top() * optix::make_float4(v, 1);
	return optix::make_float3(vh) / vh.w; 
}

optix::float3 SceneLoader::transformNormal(optix::float3 n)
{
	return optix::make_float3(this->transStack.top() * make_float4(n, 0));
}

template <class T>
bool SceneLoader::readValues(std::stringstream& s, const int numvals, T* values)
{
	for (int i = 0; i < numvals; i++)
	{
		s >> values[i];
		if (s.fail())
		{
			std::cout << "Failed reading value " << i << " will skip" << std::endl;
			return false;
		}
	}
	return true;
}


std::shared_ptr<Scene> SceneLoader::load(std::string sceneFilename)
{
	// Attempt to open the scene file 
	std::ifstream in(sceneFilename);
	if (!in.is_open())
	{
		// Unable to open the file. Check if the filename is correct.
		throw std::runtime_error("Unable to open scene file " + sceneFilename);
	}

	auto scene = std::make_shared<Scene>();

	this->transStack.push(optix::Matrix4x4::identity());

	std::string str, cmd;

	//	Temporal stack values
	//		Material
	int brdfMode = 0;
	optix::float3 ambient = optix::make_float3( 0 );
	optix::float3 specular = optix::make_float3( 0 );
	float shininess = 1;
	optix::float3 emission = optix::make_float3( 0 );
	optix::float3 diffuse = optix::make_float3( 0 );
	optix::float3 attenuation = optix::make_float3( 1, 0, 0 );
	float roughness = 1;
	int lightId = 1;

	//		Transform
	std::vector< optix::Matrix4x4 > transformStack( 1, optix::Matrix4x4::identity() );

	// Read a line in the scene file in each iteration
	while (std::getline(in, str))
	{
		// Ruled out comment and blank lines
		if ((str.find_first_not_of(" \t\r\n") == std::string::npos) 
			|| (str[0] == '#'))
		{
			continue;
		}

		// Read a command
		std::stringstream s(str);
		s >> cmd;

		// Some arrays for storing values
		float fvalues[12];
		int ivalues[3];
		std::string svalues[1];

		/*
			Scene Setting
		*/
		if (cmd == "size" && readValues(s, 2, fvalues))
		{
			scene->width = (unsigned int)fvalues[0];
			scene->height = (unsigned int)fvalues[1];
		}
		else if (cmd == "gamma" && readValues( s, 1, fvalues ))
		{
			scene->gamma = (float)fvalues[ 0 ];
		}
		else if (cmd == "integrator" && readValues( s, 1, svalues ))
		{
			scene->integratorName = svalues[ 0 ];
		}
		else if (cmd == "output" && readValues(s, 1, svalues))
		{
			scene->outputFilename = svalues[0];
		}
		else if (cmd == "maxdepth" && readValues( s, 1, ivalues ))
		{
			scene->maxdepth = (int)ivalues[ 0 ];
		}
		else if (cmd == "lightsamples" && readValues( s, 1, ivalues ))
		{
			scene->lightsamples = (int)ivalues[ 0 ];
		}
		else if (cmd == "lightstratify" && readValues( s, 1, svalues ))
		{
			if (svalues[ 0 ] == "on")
				scene->lightstratify = true;
			else if (svalues[ 0 ] == "off")
				scene->lightstratify = false;
		}
		else if (cmd == "spp" && readValues( s, 1, ivalues ))
		{
			scene->samplePerPixel = (int)ivalues[ 0 ];
		}
		else if (cmd == "nexteventestimation" && readValues( s, 1, svalues ))
		{
			if (svalues[ 0 ] == "off")
				scene->nexteventestimationMode = 0;
			else if (svalues[ 0 ] == "on")
				scene->nexteventestimationMode = 1;
			else if (svalues[ 0 ] == "mis")
				scene->nexteventestimationMode = 2;
		}
		else if (cmd == "russianroulette" && readValues( s, 1, svalues ))
		{
			if (svalues[ 0 ] == "on")
				scene->russianroulette = true;
			else if (svalues[ 0 ] == "off")
				scene->russianroulette = false;
		}
		else if (cmd == "importancesampling" && readValues( s, 1, svalues ))
		{
			if (svalues[ 0 ] == "hemisphere")
				scene->importanceSamplingMode = 0;
			else if (svalues[ 0 ] == "cosine")
				scene->importanceSamplingMode = 1;
			else if (svalues[ 0 ] == "brdf")
				scene->importanceSamplingMode = 2;
		}
		else if (cmd == "camera" && readValues( s, 10, fvalues ))
		{
			scene->camPos.x = (float)fvalues[ 0 ];
			scene->camPos.y = (float)fvalues[ 1 ];
			scene->camPos.z = (float)fvalues[ 2 ];

			scene->camLookAt.x = (float)fvalues[ 3 ];
			scene->camLookAt.y = (float)fvalues[ 4 ];
			scene->camLookAt.z = (float)fvalues[ 5 ];

			scene->camUp.x = (float)fvalues[ 6 ];
			scene->camUp.y = (float)fvalues[ 7 ];
			scene->camUp.z = (float)fvalues[ 8 ];
			
			scene->camFov = (float)fvalues[ 9 ];

			std::cout << "Done read camera!" << std::endl;

		}
		/*
			Shape
		*/
		else if (cmd == "vertex" && readValues( s, 3, fvalues ))
		{
			scene->vertices.push_back( optix::make_float3(
				( float )fvalues[ 0 ]
				, ( float )fvalues[ 1 ]
				, ( float )fvalues[ 2 ]
			));
		}
		else if (cmd == "tri" && readValues( s, 3, ivalues ))
		{
			//	Create a Triangle
			Triangle triangle(
				scene->vertices[ (unsigned int)ivalues[ 0 ] ]
				, scene->vertices[ (unsigned int)ivalues[ 1 ] ]
				, scene->vertices[ (unsigned int)ivalues[ 2 ] ]
			);
			
			//	Assign its material
			triangle.shapeMaterial.brdfMode = brdfMode;
			triangle.shapeMaterial.ambient = ambient;
			triangle.shapeMaterial.specular = specular;
			triangle.shapeMaterial.shininess = shininess;
			triangle.shapeMaterial.emission = emission;
			triangle.shapeMaterial.diffuse = diffuse;
			triangle.shapeMaterial.roughness = roughness;
			triangle.shapeMaterial.lightId = 0;

			//	Transform
			triangle.transform = transformStack.back();
			triangle.invTransform = triangle.transform.inverse();
			triangle.transposeInvTransform = triangle.invTransform.transpose();

			//	Append into list
			scene->triangles.push_back( triangle );
		}
		else if (cmd == "sphere" && readValues( s, 4, fvalues ))
		{
			//	Create a Sphere
			Sphere sphere(
				optix::make_float3(
					(float)fvalues[ 0 ]
					, (float)fvalues[ 1 ]
					, (float)fvalues[ 2 ]
				)
				, (float)fvalues[ 3 ]
			);

			//	Assign its material
			sphere.shapeMaterial.brdfMode = brdfMode;
			sphere.shapeMaterial.ambient = ambient;
			sphere.shapeMaterial.specular = specular;
			sphere.shapeMaterial.shininess = shininess;
			sphere.shapeMaterial.emission = emission;
			sphere.shapeMaterial.diffuse = diffuse;
			sphere.shapeMaterial.roughness = roughness;
			sphere.shapeMaterial.lightId = 0;

			//	Transform
			sphere.transform = transformStack.back();
			sphere.invTransform = sphere.transform.inverse();
			sphere.transposeInvTransform = sphere.invTransform.transpose();

			//	Append into list
			scene->spheres.push_back( sphere );
		}
		/*
			Material
		*/
		else if (cmd == "brdf" && readValues( s, 1, svalues ))
		{
			if (svalues[ 0 ] == "phong")
				brdfMode = 0;
			else if (svalues[ 0 ] == "ggx")
				brdfMode = 1;
		}
		else if (cmd == "ambient" && readValues( s, 3, fvalues ))
		{
			ambient.x = (float)fvalues[ 0 ];
			ambient.y = (float)fvalues[ 1 ];
			ambient.z = (float)fvalues[ 2 ];
		}
		else if (cmd == "emission" && readValues( s, 3, fvalues ))
		{
			emission.x = (float)fvalues[ 0 ];
			emission.y = (float)fvalues[ 1 ];
			emission.z = (float)fvalues[ 2 ];
		}
		else if (cmd == "diffuse" && readValues( s, 3, fvalues ))
		{
			diffuse.x = (float)fvalues[ 0 ];
			diffuse.y = (float)fvalues[ 1 ];
			diffuse.z = (float)fvalues[ 2 ];
		}
		else if (cmd == "specular" && readValues( s, 3, fvalues ))
		{
			specular.x = (float)fvalues[ 0 ];
			specular.y = (float)fvalues[ 1 ];
			specular.z = (float)fvalues[ 2 ];
		}
		else if (cmd == "shininess" && readValues( s, 1, fvalues ))
		{
			shininess = (float)fvalues[ 0 ];
		}
		else if (cmd == "attenuation" && readValues( s, 3, fvalues ))
		{
			attenuation.x = (float)fvalues[ 0 ];
			attenuation.y = (float)fvalues[ 1 ];
			attenuation.z = (float)fvalues[ 2 ];
		}
		else if (cmd == "roughness" && readValues( s, 1, fvalues ))
		{
			roughness = (float)fvalues[ 0 ];
		}
		/*
			Transform
		*/
		else if (cmd == "popTransform" && readValues( s, 0, svalues ))
		{
			transformStack.pop_back();
		}
		else if (cmd == "pushTransform" && readValues( s, 0, svalues ))
		{
			transformStack.push_back( transformStack.back() );
		}
		else if (cmd == "translate" && readValues( s, 3, fvalues ))
		{
			transformStack.back() = transformStack.back() * optix::Matrix4x4::translate( optix::make_float3(
				(float)fvalues[ 0 ]
				, (float)fvalues[ 1 ]
				, (float)fvalues[ 2 ]
			) );
		}
		else if (cmd == "rotate" && readValues( s, 4, fvalues ))
		{
			transformStack.back() = transformStack.back() * optix::Matrix4x4::rotate(
				(float)fvalues[ 3 ] * ( M_PIf / 180.f )
				, optix::make_float3(
					(float)fvalues[ 0 ]
					, (float)fvalues[ 1 ]
					, (float)fvalues[ 2 ]
				)
			);
		}
		else if (cmd == "scale" && readValues( s, 3, fvalues ))
		{
			transformStack.back() = transformStack.back() * optix::Matrix4x4::scale( optix::make_float3(
				(float)fvalues[ 0 ]
				, (float)fvalues[ 1 ]
				, (float)fvalues[ 2 ]
			) );
		}
		/*
			Light
		*/
		else if (cmd == "point" && readValues( s, 6, fvalues ))
		{
			//	Create a PointLight 
			PointLight pointLight;
			pointLight.position.x = (float)fvalues[ 0 ];
			pointLight.position.y = (float)fvalues[ 1 ];
			pointLight.position.z = (float)fvalues[ 2 ];
			pointLight.color.x = (float)fvalues[ 3 ];
			pointLight.color.y = (float)fvalues[ 4 ];
			pointLight.color.z = (float)fvalues[ 5 ];
			pointLight.attenuation = attenuation;

			//	Append into list
			scene->plights.push_back( pointLight );
		}
		else if (cmd == "directional" && readValues( s, 6, fvalues ))
		{
			//	Create a PointLight 
			DirectionalLight directionalLight;
			directionalLight.direction.x = (float)fvalues[ 0 ];
			directionalLight.direction.y = (float)fvalues[ 1 ];
			directionalLight.direction.z = (float)fvalues[ 2 ];
			directionalLight.direction = optix::normalize( directionalLight.direction );

			directionalLight.color.x = (float)fvalues[ 3 ];
			directionalLight.color.y = (float)fvalues[ 4 ];
			directionalLight.color.z = (float)fvalues[ 5 ];

			//	Append into list
			scene->dlights.push_back( directionalLight );
		}
		else if (cmd == "quadLight" && readValues( s, 12, fvalues ))
		{
			QuadLight quadLight;
			quadLight.a.x = (float)fvalues[ 0 ];
			quadLight.a.y = (float)fvalues[ 1 ];
			quadLight.a.z = (float)fvalues[ 2 ];

			quadLight.ab.x = (float)fvalues[ 3 ];
			quadLight.ab.y = (float)fvalues[ 4 ];
			quadLight.ab.z = (float)fvalues[ 5 ];

			quadLight.ac.x = (float)fvalues[ 6 ];
			quadLight.ac.y = (float)fvalues[ 7 ];
			quadLight.ac.z = (float)fvalues[ 8 ];

			quadLight.intensity.x = (float)fvalues[ 9 ];
			quadLight.intensity.y = (float)fvalues[ 10 ];
			quadLight.intensity.z = (float)fvalues[ 11 ];

			quadLight.lightId = lightId;

			//	NOTE: The input file here is intentionally set ab and ac such that  ab x ac is in
			//		the opposite direction of the intended light. BECAUSE it might be easier when
			//		calculating the Geometric term in Monte-Carlo direct light.
			//		BUT I do NOT like it. So I'm going to invert it back the other way around
			//		because I found that it easier to understand intuitively. Hence, the notation minus(-)
			quadLight.normal = -optix::normalize( optix::cross( quadLight.ab, quadLight.ac ) );

			//	Append into the list
			scene->qLights.push_back( quadLight );

			//	Also create a quad plane of emission as intensity
			Triangle triangle1(
				quadLight.a
				, quadLight.a + quadLight.ab
				, quadLight.a + quadLight.ac
			);

			Triangle triangle2(
				quadLight.a + quadLight.ab
				, quadLight.a + quadLight.ab + quadLight.ac
				, quadLight.a + quadLight.ac
			);

			//	Assign its material
			triangle2.shapeMaterial.brdfMode = triangle1.shapeMaterial.brdfMode = 0;
			triangle2.shapeMaterial.ambient  = triangle1.shapeMaterial.ambient = optix::make_float3( 0 );
			triangle2.shapeMaterial.specular = triangle1.shapeMaterial.specular = optix::make_float3( 0 );
			triangle2.shapeMaterial.shininess = triangle1.shapeMaterial.shininess = 1;
			triangle2.shapeMaterial.emission = triangle1.shapeMaterial.emission = quadLight.intensity;
			triangle2.shapeMaterial.diffuse  = triangle1.shapeMaterial.diffuse = optix::make_float3( 0 );
			triangle2.shapeMaterial.roughness = triangle1.shapeMaterial.roughness = 1;
			triangle2.shapeMaterial.lightId = triangle1.shapeMaterial.lightId = lightId;

			lightId++;

			//	Transform
			triangle2.transform = triangle1.transform = optix::Matrix4x4::identity();
			triangle2.invTransform = triangle1.invTransform = triangle1.transform.inverse();
			triangle2.transposeInvTransform = triangle1.transposeInvTransform = triangle1.invTransform.transpose();

			//	Append into list
			scene->triangles.push_back( triangle1 );
			scene->triangles.push_back( triangle2 );
		}
	}

	in.close();

	return scene;
}