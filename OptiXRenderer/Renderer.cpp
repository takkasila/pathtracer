#include "Renderer.h"

#include <ctime>

using namespace optix;

Renderer::Renderer(std::shared_ptr<Scene> scene, bool isProgressive ) 
	:	scene(scene)
		, isProgressive( isProgressive )
{
	// Create the Optix context
	this->context = Context::create();

	// Configure the context 
	this->context->setRayTypeCount(2); // two types of rays: normal and shadow rays
	this->context->setEntryPointCount(1); // only one entry point
	this->context->setPrintEnabled(true); // enable the use of rtPrintf in programs
	this->context->setPrintBufferSize(2048); 
	this->context->setMaxTraceDepth(3); // Set maximum recursion depth.

	// Create the resultBuffer
	this->resultBuffer = this->context->createBuffer(RT_BUFFER_OUTPUT); // only device can write
	this->resultBuffer->setFormat(RT_FORMAT_FLOAT3); // each entry is of type float3

	// Initialize Optix programs
	initPrograms();

	// Build the scene by constructing an Optix graph
	buildScene();
}

void Renderer::run()
{
	while (this->currentFrame != this->numFrames)
	{
		std::cout << "Renderer::run()" << std::endl;
		// Render a frame.
		this->context["frameID"]->setInt(++this->currentFrame);

		//	A simple timer
		clock_t startTime = clock();

		this->context->launch(0, this->width, this->height);

		clock_t endTime = clock();
		double useTime = double(endTime - startTime) / CLOCKS_PER_SEC;
		std::cout << "sec per frame: " << useTime << std::endl;

		// Only render a frame in progressive mode
		if ( this->isProgressive ) 
			break;
	}
}

void Renderer::initPrograms()
{
	// Ray generation program
	this->programs[ "rayGen" ] = this->createProgram( "PinholeCamera.cu", "generateRays" );
	this->context->setRayGenerationProgram( 0, this->programs[ "rayGen" ] );

	// Miss progarm
	this->programs[ "miss" ] = this->createProgram( "Common.cu", "miss" );
	this->programs[ "miss" ][ "backgroundColor" ]->setFloat( 0, 0, 0 );
	this->context->setMissProgram( 0, this->programs[ "miss" ] );

	// Exception program
	this->programs[ "exc" ] = this->createProgram( "Common.cu", "exception" );
	this->context->setExceptionEnabled( RT_EXCEPTION_ALL, true );
	this->context->setExceptionProgram( 0, this->programs[ "exc" ] );

	// Triangle programs
	this->programs[ "triInt" ] = this->createProgram( "Triangle.cu", "intersect" );
	this->programs[ "triBound" ] = this->createProgram( "Triangle.cu", "bound" );

	// Sphere programs 
	this->programs[ "sphereInt" ] = this->createProgram( "Sphere.cu", "intersect" );
	this->programs[ "sphereBound" ] = this->createProgram( "Sphere.cu", "bound" );

	// Integrators
	this->programs[ "pathtracer" ] = this->createProgram( "PathTracer.cu", "closestHit_pathtracer" );
	
	// Shadow Caster
	this->programs[ "shadowCaster_anyHit" ] = this->createProgram( "Common.cu", "shadowCaster_anyHit" );
	this->programs[ "shadowCaster_closestHit" ] = this->createProgram( "Common.cu", "shadowCaster_closestHit" );

}

std::vector<unsigned char> Renderer::getResult()
{
	// Cast a float number (0 to 1) to a byte (0 to 255)
	auto cast = [](float v)
	{
		v = v > 1.f ? 1.f : v < 0.f ? 0.f : v;
		return static_cast<unsigned char>(v * 255);
	};

	float3* bufferData = (float3*)this->resultBuffer->map();

	// Store the data into a byte vector
	std::vector<unsigned char> imageData(this->width * this->height * 4);
	for (int i = 0; i < this->height; i++)
	{
		for (int j = 0; j < this->width; j++)
		{
			int index = (i * this->width + j) * 4;
			float3 pixel = bufferData[(this->height - i - 1) * this->width + j];
			imageData[index + 0] = cast(pixel.x);
			imageData[index + 1] = cast(pixel.y);
			imageData[index + 2] = cast(pixel.z);
			imageData[index + 3] = 255; // alpha channel
		}
	}

	this->resultBuffer->unmap();

	return imageData;
}

Program Renderer::createProgram(const std::string& filename, 
	const std::string& programName)
{
	const char* ptx = sutil::getPtxString("OptiXRenderer", filename.c_str());
	return this->context->createProgramFromPTXString(ptx, programName);
}

template <class T>
Buffer Renderer::createBuffer(std::vector<T> data)
{
	Buffer buffer = this->context->createBuffer(RT_BUFFER_INPUT); // only host can write
	buffer->setFormat(RT_FORMAT_USER); // use user-defined format
	buffer->setElementSize(sizeof(T)); // size of an element
	buffer->setSize(data.size()); // number of elements
	std::memcpy(buffer->map(), data.data(), sizeof(T) * data.size());
	buffer->unmap();
	return buffer;
}

void Renderer::buildScene()
{
	// Record some important info
	this->width = this->scene->width;
	this->height = this->scene->height;
	this->outputFilename = this->scene->outputFilename;
	this->currentFrame = 0;
	this->numFrames = 1;

	// Set width and height
	this->resultBuffer->setSize(this->width, this->height);
	this->programs["rayGen"]["resultBuffer"]->set(resultBuffer);
	this->context["width"]->setFloat(this->width);
	this->context["height"]->setFloat(this->height);

	// Set material programs based on integrator type.
	this->programs["integrator"] = this->programs[this->scene->integratorName];
	Material material = this->context->createMaterial();
	material->setClosestHitProgram(0, this->programs["integrator"]);
	material->setAnyHitProgram(1, this->programs["shadowCaster_anyHit"]);
	material->setClosestHitProgram( 1, this->programs[ "shadowCaster_closestHit" ] );


	//	Pass Scene's camera data to rayGen program
	this->programs[ "rayGen" ][ "camPos" ]->set3fv( (float*)&this->scene->camPos );
	this->programs[ "rayGen" ][ "camLookAt" ]->set3fv( (float*)&this->scene->camLookAt );
	this->programs[ "rayGen" ][ "camUp" ]->set3fv( (float*)&this->scene->camUp );
	this->programs[ "rayGen" ][ "camFov" ]->set1fv( &this->scene->camFov );

	//	Pass Scene data to rayGen program
	unsigned int imgSize[] = {this->width, this->height};
	this->programs[ "rayGen" ][ "imgSize" ]->set2uiv( imgSize );

	float gamma = this->scene->gamma;
	this->programs[ "rayGen" ][ "gamma" ]->set1fv( &gamma );

	int maxdepth = this->scene->maxdepth;
	this->programs[ "rayGen" ][ "maxdepth" ]->set1iv( &maxdepth );

	int samplePerPixel = this->scene->samplePerPixel;
	this->programs[ "rayGen" ][ "samplePerPixel" ]->set1iv( &samplePerPixel );

	int russianroulette = this->scene->russianroulette;
	this->programs[ "rayGen" ][ "russianroulette" ]->set1iv( &russianroulette );

	// Create buffers and pass them to Optix programs that the buffers
	Buffer triBuffer = this->createBuffer(this->scene->triangles);
	this->programs["triInt"]["triangles"]->set(triBuffer);
	this->programs["triBound"]["triangles"]->set(triBuffer);

	Buffer sphereBuffer = this->createBuffer(this->scene->spheres);
	this->programs["sphereInt"]["spheres"]->set(sphereBuffer);
	this->programs["sphereBound"]["spheres"]->set(sphereBuffer);

	// Construct the Optix graph. It should look like:
	//	  root
	//	   ||
	//	   GG
	//	//	\\
	//   triGI   sphereGI
	//  //  \\	//   \\
	// triGeo material sphereGeo

	Geometry triGeo = this->context->createGeometry();
	triGeo->setPrimitiveCount(this->scene->triangles.size());
	triGeo->setIntersectionProgram(this->programs["triInt"]);
	triGeo->setBoundingBoxProgram(this->programs["triBound"]);

	Geometry sphereGeo = this->context->createGeometry();
	sphereGeo->setPrimitiveCount(this->scene->spheres.size());
	sphereGeo->setIntersectionProgram(this->programs["sphereInt"]);
	sphereGeo->setBoundingBoxProgram(this->programs["sphereBound"]);

	GeometryInstance triGI = this->context->createGeometryInstance();
	triGI->setGeometry(triGeo);
	triGI->setMaterialCount(1);
	triGI->setMaterial(0, material);

	GeometryInstance sphereGI = this->context->createGeometryInstance();
	sphereGI->setGeometry(sphereGeo);
	sphereGI->setMaterialCount(1);
	sphereGI->setMaterial(0, material);

	GeometryGroup GG = this->context->createGeometryGroup();
	GG->setAcceleration(this->context->createAcceleration("Trbvh"));
	GG->setChildCount(1);
	GG->setChild(0, sphereGI);

	GeometryGroup triGG = this->context->createGeometryGroup();
	triGG->setAcceleration(this->context->createAcceleration("Trbvh"));
	triGG->setChildCount(1);
	triGG->setChild(0, triGI);

	Group root = this->context->createGroup();
	root->setAcceleration(this->context->createAcceleration("Trbvh"));
	root->setChildCount(2);
	root->setChild(0, triGG);
	root->setChild(1, GG);
	this->programs["rayGen"]["root"]->set(root);
	this->programs["integrator"]["root"]->set(root);

	// Create buffers for lights
	Buffer plightBuffer = this->createBuffer( this->scene->plights );
	this->programs[ "integrator" ][ "plights" ]->set( plightBuffer );
	Buffer dlightBuffer = this->createBuffer( this->scene->dlights );
	this->programs[ "integrator" ][ "dlights" ]->set( dlightBuffer );
	Buffer qlightBuffer = this->createBuffer( this->scene->qLights );
	this->programs[ "integrator" ][ "qlights" ]->set( qlightBuffer );

	int numPointLight = this->scene->plights.size();
	int numDirectionalLight = this->scene->dlights.size();
	int numQuadLight = this->scene->qLights.size();
	int numLightSample = this->scene->lightsamples;
	int isStratify = this->scene->lightstratify;
	int nexteventestimationMode = this->scene->nexteventestimationMode;
	int importanceSamplingMode = this->scene->importanceSamplingMode;
	this->programs[ "integrator" ][ "numPointLight" ]->set1iv( &numPointLight );
	this->programs[ "integrator" ][ "numDirectionalLight" ]->set1iv( &numDirectionalLight );
	this->programs[ "integrator" ][ "numQuadLight" ]->set1iv( &numQuadLight );
	this->programs[ "integrator" ][ "numLightSample" ]->set1iv( &numLightSample );
	this->programs[ "integrator" ][ "isStratify" ]->set1iv( &isStratify );
	this->programs[ "integrator" ][ "nexteventestimationMode" ]->set1iv( &nexteventestimationMode );
	this->programs[ "integrator" ][ "importanceSamplingMode" ]->set1iv( &importanceSamplingMode );

	// Validate everything before running 
	this->context->validate();
}