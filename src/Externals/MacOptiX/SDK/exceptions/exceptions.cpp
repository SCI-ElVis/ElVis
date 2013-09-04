
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

/*
 * exceptions.cpp -- Demonstrates use of exceptions and exception codes.
 */

#include <optix.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sutil.h>
#include <string>
#include <iostream>


void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"

    << "  -h  | --help                               Print this usage message\n"
    << std::endl;

  if ( doExit ) exit(1);
}


int main(int argc, char* argv[])
{
    /* Primary OptiX objects */
    RTcontext          context;
    RTprogram          ray_gen_program;
    RTprogram          exception_program;
    RTprogram          miss_program;
    RTbuffer           buffer;
    RTselector         dummy_selector;
    RTgeometrygroup    dummy_group;
    RTvariable         dummy_object;
    RTacceleration     dummy_acceleration;
    RTprogram          selector_program;
    RTmaterial         material;
    RTprogram          closest_hit_program;
    RTprogram          any_hit_program;
    RTgeometryinstance instance;
    RTgeometry         geometry;
    RTprogram          intersection_program;
    RTprogram          bounding_box_program;

    /* Parameters */
    RTvariable result_buffer;

    char path_to_ptx[512];
    char outfile[512];

    unsigned int width  = 512u;
    unsigned int height = 384u;
    int i;

    outfile[0] = '\0';

    /* Process command line args */
    RT_CHECK_ERROR_NO_CONTEXT( sutilInitGlut( &argc, argv ) );
    for( i = 1; i < argc; ++i ) {
      if( strcmp( argv[i], "--help" ) == 0 || strcmp( argv[i], "-h" ) == 0 ) {
        printUsageAndExit( argv[0] );
      } else {
        fprintf( stderr, "Unknown option '%s'\n", argv[i] );
        printUsageAndExit( argv[0] );
      }
    }

    /* Create our objects and set state */
    RT_CHECK_ERROR( rtContextCreate( &context ) );
    RT_CHECK_ERROR( rtContextSetRayTypeCount( context, 1 ) );
    RT_CHECK_ERROR( rtContextSetEntryPointCount( context, 1 ) );

    RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_OUTPUT, &buffer ) );
    RT_CHECK_ERROR( rtBufferSetFormat( buffer, RT_FORMAT_FLOAT4 ) );
    RT_CHECK_ERROR( rtBufferSetSize2D( buffer, width, height ) );
    RT_CHECK_ERROR( rtContextDeclareVariable( context, "result_buffer", &result_buffer ) );
    RT_CHECK_ERROR( rtVariableSetObject( result_buffer, buffer ) );

    sprintf( path_to_ptx, "%s/%s", sutilSamplesPtxDir(), "exceptions_generated_exceptions_programs.cu.ptx" );
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "buggy_draw_solid_color", &ray_gen_program ) );
    RT_CHECK_ERROR( rtContextSetRayGenerationProgram( context, 0, ray_gen_program ) );

    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "exception", &exception_program ) );
    RT_CHECK_ERROR( rtContextSetExceptionProgram( context, 0, exception_program ) );

    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "non_terminating_miss_program", &miss_program ) );
    RT_CHECK_ERROR( rtContextSetMissProgram( context, 0, miss_program ) );

    RT_CHECK_ERROR( rtSelectorCreate( context, &dummy_selector ) );
    RT_CHECK_ERROR( rtGeometryGroupCreate( context, &dummy_group ) );
    RT_CHECK_ERROR( rtSelectorSetChildCount( dummy_selector, 1 ) );
    RT_CHECK_ERROR( rtSelectorSetChild( dummy_selector, 0, dummy_group ) );

    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "visit", &selector_program ) );
    RT_CHECK_ERROR( rtSelectorSetVisitProgram( dummy_selector, selector_program ) );
    RT_CHECK_ERROR( rtContextDeclareVariable( context, "dummy_object", &dummy_object ) );
    RT_CHECK_ERROR( rtVariableSetObject( dummy_object, dummy_selector ) );

    RT_CHECK_ERROR( rtAccelerationCreate( context, &dummy_acceleration ) );
    RT_CHECK_ERROR( rtAccelerationSetBuilder( dummy_acceleration,"NoAccel") );
    RT_CHECK_ERROR( rtAccelerationSetTraverser( dummy_acceleration,"NoAccel") );
    RT_CHECK_ERROR( rtGeometryGroupSetAcceleration( dummy_group, dummy_acceleration) );


    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "closest_hit", &closest_hit_program ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "any_hit", &any_hit_program ) );
    RT_CHECK_ERROR( rtMaterialCreate( context, &material ) );
    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( material, 0, closest_hit_program ) );
    RT_CHECK_ERROR( rtMaterialSetAnyHitProgram( material, 0, any_hit_program ) );

    RT_CHECK_ERROR( rtGeometryInstanceCreate( context, &instance ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetMaterialCount( instance, 1 ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( instance, 0, material ) );

    RT_CHECK_ERROR( rtGeometryCreate( context, &geometry ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetGeometry( instance, geometry ) );
    RT_CHECK_ERROR( rtGeometryGroupSetChildCount( dummy_group, 1 ) );
    RT_CHECK_ERROR( rtGeometryGroupSetChild( dummy_group, 0, instance ) );
    RT_CHECK_ERROR( rtGeometrySetPrimitiveCount( geometry, 1 ) );

    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "bounds", &bounding_box_program ) );
    RT_CHECK_ERROR( rtGeometrySetBoundingBoxProgram( geometry, bounding_box_program ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "intersect", &intersection_program ) );
    RT_CHECK_ERROR( rtGeometrySetIntersectionProgram( geometry, intersection_program ) );


    /* Enable checking for all exceptions */
    RT_CHECK_ERROR( rtContextSetExceptionEnabled( context, RT_EXCEPTION_ALL, 1 ) );

    /* Enable printing so rtPrintExceptionDetails has an effect */
    RT_CHECK_ERROR( rtContextSetPrintEnabled( context, 1 ) );

    /* Run */
    RT_CHECK_ERROR( rtContextValidate( context ) );
    RT_CHECK_ERROR( rtContextCompile( context ) );
    RT_CHECK_ERROR( rtContextLaunch2D( context, 0 /* entry point */, width, height ) );

    /* Display image */
    if( strlen( outfile ) == 0 ) {
      RT_CHECK_ERROR( sutilDisplayBufferInGlutWindow( argv[0], buffer ) );
    } else {
      RT_CHECK_ERROR( sutilDisplayFilePPM( outfile, buffer ) );
    }

    /* Clean up */
    RT_CHECK_ERROR( rtBufferDestroy( buffer ) );
    RT_CHECK_ERROR( rtProgramDestroy( ray_gen_program ) );
    RT_CHECK_ERROR( rtContextDestroy( context ) );

    return( 0 );
}

