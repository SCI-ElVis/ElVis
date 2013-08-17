
/*
 * Copyright (c) 2010 NVIDIA Corporation.  All rights reserved.
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

//-------------------------------------------------------------------------------
//
//  zoneplate.cpp -- aliasing torture test 
//
//-------------------------------------------------------------------------------



#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  if defined(_WIN32)
#    include <GL/wglew.h>
#  endif

#  include <GL/glut.h>
#endif

#include "zoneplate_util.h"
#include "zoneplate_common.h"

#include <SampleScene.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>

using namespace optix;

unsigned int   window_width = 512;
unsigned int   window_height = 512;
Context        m_context;
bool           zoneplate_computed = false;
int            sqrt_samples_per_pixel = 3;
float          filter_width = 1.0f;
float          gaussian_alpha = 1.0f;
float          sinc_tau = 1.0f;
float          checkerboard_rotate = 30.0f;
int            checkerboard_width = 200;
int            contrast_window_width = 1;
float          adaptive_contrast_threshold = .8f;
float          jitter_amount = 0.0f;
bool           text_overlay = true;
bool           debug_display = false;
bool           motion = false;

double         last_frame_time       = 0.0;
unsigned int   last_frame_count      = 0;
unsigned int   frame_count           = 0;
double         fps_update_threshold  = 0.5;

enum { 
  ENTRY_COLOR_ONLY = 0,
  ENTRY_GENERATE_SAMPLES,
  ENTRY_GATHER_SAMPLES,
  ENTRY_FLOAT_ATOMICS,
  ENTRY_ZERO_SCATTER_BUFFERS,
  ENTRY_SCATTER_DO_DIVIDE,
  ENTRY_FIND_CONTRAST_LOCATIONS,
  ENTRY_ADAPTIVE_RESAMPLE,
  NUM_ENTRY_POINTS
};

zpFilterType filter_type = FILTER_BOX;
zpRenderType render_type = RENDER_CHECKERBOARD;
zpAAType aa_type = AA_NONE;

void createOptiXBuffers()
{
  Buffer color_buffer;
  Buffer sample_buffer;
  Buffer filter_weights;
  Buffer weighted_scatter_sums;
  Buffer adaptive_sample_locations;

  color_buffer = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, window_width, window_height );
  sample_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, sqrt_samples_per_pixel * window_width, sqrt_samples_per_pixel * window_height );
  sample_buffer->setElementSize( sizeof( zpSample )) ;
  filter_weights = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, window_width, window_height );
  weighted_scatter_sums = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, window_width, window_height );
  adaptive_sample_locations = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_BYTE, window_width, window_height );

  m_context["output_color_only"]->set(color_buffer);
  m_context["adaptive_sample_locations"]->set(adaptive_sample_locations);
  m_context["output_samples"]->set(sample_buffer); // need all this so the compile will function.
  m_context["filter_weights"]->set(filter_weights);
  m_context["weighted_scatter_sums"]->set(weighted_scatter_sums);
}

void resizeOptiXBuffers()
{
  m_context["output_color_only"]->getBuffer()->setSize( window_width, window_height );
  switch( aa_type ) {
    case AA_NONE:
      break;
    case AA_SAMPLE_GATHER:
      m_context["output_samples"]->getBuffer()->setSize( sqrt_samples_per_pixel * window_width, sqrt_samples_per_pixel * window_height );
      break;
    case AA_CONTRAST_THRESHOLD:
      m_context["adaptive_sample_locations"]->getBuffer()->setSize( window_width, window_height );
      // fallthrough intentional
    case AA_FLOAT_ATOMICS:
      m_context["filter_weights"]->getBuffer()->setSize( window_width, window_height );
      m_context["weighted_scatter_sums"]->getBuffer()->setSize( window_width, window_height );
      break;
    default:
      std::cerr << "WARNING: UNKNOWN aa_type IN INIT BUFFERS" << std::endl;
      break;
  }
}

Buffer getOutputBuffer()
{
  return m_context["output_color_only"]->getBuffer();
}

Buffer getAdaptiveSampleLocationsBuffer()
{
  return m_context["adaptive_sample_locations"]->getBuffer();
}


void keyboard(unsigned char key, int x, int y)
{
  switch( key ) {
    case 27:
    case 'q':
    case 'Q':
      m_context->destroy();
      exit(0);
    case 'f':
      filter_type = (zpFilterType)((int)filter_type + 1);
      if (filter_type == NUM_FILTERS) filter_type = FILTER_BOX;
      setFilterType( filter_type );
      glutPostRedisplay();
      break;
    case 'w':
      setFilterWidth( filter_width - .5f );
      break;
    case 'W':
      setFilterWidth( filter_width + .5f );
      break;
    case 's':
      setSQPP( sqrt_samples_per_pixel - 1 );
      break;
    case 'S':
      setSQPP( sqrt_samples_per_pixel + 1 );
      break;
    case 'c':
      setCheckerboardWidth( checkerboard_width -1 );
      break;
    case 'C':
      setCheckerboardWidth( checkerboard_width + 1 );
      break;
    case 'a':
      setAdaptiveContrastThreshold( adaptive_contrast_threshold - 0.05f );
      break;
    case 'A':
      setAdaptiveContrastThreshold( adaptive_contrast_threshold + 0.05f );
      break;
    case 'r':
      motion = !motion;
      glutPostRedisplay();
      break;
    case 'd':
      debug_display = !debug_display;
      glutPostRedisplay();
      break;
    case 'm':
      aa_type = (zpAAType)((int)aa_type + 1);
      if (aa_type == NUM_AA_TYPES) aa_type = AA_NONE;
      setAAType( aa_type );
      glutPostRedisplay();
      break;
    case 'p':
      render_type = (zpRenderType)((int)render_type + 1);
      if (render_type == NUM_RENDER_TYPES) render_type = RENDER_ZONEPLATE;
      setRenderType( render_type );
      glutPostRedisplay();
      break;
    case 't':
      text_overlay = !text_overlay;
      glutPostRedisplay();
      break;
  }
}

void special( int key, int x, int y )
{
  switch( key ) {
    case GLUT_KEY_LEFT:
       setCheckerboardRotate( checkerboard_rotate - 5 );
      break;
    case GLUT_KEY_RIGHT:
      setCheckerboardRotate( checkerboard_rotate + 5 );
      break;
  }
}


void recompute_zoneplate() {
  switch( aa_type ) {
    case AA_NONE:
      m_context->launch(ENTRY_COLOR_ONLY, window_width, window_height );
      break;
    case AA_SAMPLE_GATHER:
      m_context->launch(ENTRY_GENERATE_SAMPLES, sqrt_samples_per_pixel * window_width, sqrt_samples_per_pixel * window_height );
      m_context->launch(ENTRY_GATHER_SAMPLES, window_width, window_height );
      break;
    case AA_FLOAT_ATOMICS:
      m_context->launch(ENTRY_ZERO_SCATTER_BUFFERS, window_width, window_height );
      m_context->launch(ENTRY_FLOAT_ATOMICS, window_width, window_height );
      m_context->launch(ENTRY_SCATTER_DO_DIVIDE, window_width, window_height );
      break;
    case AA_CONTRAST_THRESHOLD:
      {
        float save_fw = filter_width;
        setFilterWidth( 0.5 ); // force this, wider filters don't really work well
        int save_sqpp = sqrt_samples_per_pixel;
        m_context->launch(ENTRY_ZERO_SCATTER_BUFFERS, window_width, window_height );
        setSQPP( 1 );
        setJitterAmount( 0 );
        m_context->launch(ENTRY_FLOAT_ATOMICS, window_width, window_height );
        m_context->launch(ENTRY_SCATTER_DO_DIVIDE, window_width, window_height );
        m_context->launch(ENTRY_FIND_CONTRAST_LOCATIONS, window_width, window_height );
        setSQPP( save_sqpp );
        setJitterAmount( jitter_amount );
        m_context->launch(ENTRY_ADAPTIVE_RESAMPLE, window_width, window_height );
        m_context->launch(ENTRY_SCATTER_DO_DIVIDE, window_width, window_height );
        setFilterWidth( save_fw );
      }
      break;
    default:
      std::cerr << "UNIMPLEMENTED AA TYPE IN RECOMPUTE_ZONEPLATE: " << aa_type << std::endl;
   }
}

void updateScene( void )
{
  if (motion)
  {
    switch( render_type )
    {
    case RENDER_CHECKERBOARD:
      setCheckerboardRotate( checkerboard_rotate + 1 );
      break;
    default:
      std::cerr << "WARNING: NO MOTION AVAILABLE FOR RENDER TYPE " << getImageName() << std::endl;
      break;
    }
  }
}


void display( void ) 
{
  ++frame_count;

  glClearColor( 1,0,0,0 );
  glClear( GL_COLOR_BUFFER_BIT );

  if (!zoneplate_computed) {
    recompute_zoneplate();
    zoneplate_computed = true;
  }

  Buffer buffer;
  GLvoid* bufferData;

  GLenum gl_data_type;
  GLenum gl_format;

  if (!debug_display)
  {
    buffer = getOutputBuffer();
    bufferData = buffer->map();

    gl_data_type = GL_UNSIGNED_BYTE;
    gl_format = GL_BGRA;
  }
  else {
    buffer = getAdaptiveSampleLocationsBuffer();
    bufferData = buffer->map();

    gl_data_type = GL_UNSIGNED_BYTE;
    gl_format = GL_LUMINANCE;
  }


  glDrawPixels( static_cast<GLsizei>( window_width),
    static_cast<GLsizei>( window_height ),
    gl_format, gl_data_type, bufferData);

  buffer->unmap();

  static char txt[1024];
  static char fps_txt[1024];
  int line_height = 13;
  int line_spacing = 3;

  int line_gap = line_height + line_spacing;
  int y = window_height - line_gap;

#define PRINT(str) if (text_overlay) drawText( str, 10, (float) y, GLUT_BITMAP_HELVETICA_18 ); y -= line_gap

  if ( motion )
  {
    // Output fps 
    double current_time;
    sutilCurrentTime( &current_time );
    double dt = current_time - last_frame_time;
    if( dt > fps_update_threshold )
    {
      sprintf( fps_txt, "FPS: %7.2f", (frame_count - last_frame_count) / dt );

      last_frame_time = current_time;
      last_frame_count = frame_count;
    } else if( frame_count == 1 ) {
      sprintf( fps_txt, "FPS: %7.2f", 0.f );
    }
    PRINT( fps_txt );
  }

  switch( render_type ) {
    case RENDER_CHECKERBOARD:
      sprintf( txt, "Rendering: %s (width %d, rotated %f)", getImageName(), checkerboard_width, checkerboard_rotate );
      break;
    default:
      sprintf( txt, "Rendering: %s", getImageName() );
      break;
  }
  PRINT( txt );

  sprintf( txt, "Antialiasing type: %s", getAAName() );
  PRINT( txt );

  switch( aa_type )
  {
  case AA_NONE:
    break;
  case AA_CONTRAST_THRESHOLD:
    sprintf( txt, "Contrast threshold: %f", adaptive_contrast_threshold );
    PRINT( txt );
    // fallthrough intentional
  case AA_SAMPLE_GATHER:
  case AA_FLOAT_ATOMICS:
    sprintf( txt, "Filter: %s", getFilterName() );
    PRINT( txt );
    if (filter_type == FILTER_GAUSSIAN) {
      sprintf( txt, "Gaussian alpha: %f", gaussian_alpha );
      PRINT( txt );
    }
    if (filter_type == FILTER_SINC) {
      sprintf( txt, "Sinc tau: %f", sinc_tau );
      PRINT( txt );
    }
    sprintf( txt, "Filter width: %f%s", (aa_type == AA_CONTRAST_THRESHOLD ? 0.5f : filter_width), (aa_type == AA_CONTRAST_THRESHOLD ? " (forced)" : "") );
    PRINT( txt );
    sprintf( txt, "Samples per%spixel: %d", (aa_type == AA_CONTRAST_THRESHOLD ? " above threshold pixel " : " "), sqrt_samples_per_pixel * sqrt_samples_per_pixel );
    PRINT( txt );
    break;
  default:
    sprintf( txt, "UNKNOWN aa_type IN DISPLAY: %s", getAAName() );
    PRINT( txt );
  }
  glutSwapBuffers();

  updateScene();
}

void updateOptiXWindowSize() {
  static int old_w = -1, old_h = -1;

  if (old_w == static_cast<int>( window_width ) && old_h == static_cast<int>( window_height ) ) return;

  old_w = window_width;
  old_h = window_height;

  unsigned int window_size[2];
  window_size[0] = window_width;
  window_size[1] = window_height;
  m_context["window_size"]->set2uiv( window_size );
  resizeOptiXBuffers();
}

void setCameraProgram()
{
  // Pinhole Camera ray gen and exception program
  std::string         ptx_path = SampleScene::ptxpath( "zoneplate", "zoneplate.cu" );

  for (int i = 0 ; i < NUM_ENTRY_POINTS ; i++) {
    m_context->setExceptionProgram( i, m_context->createProgramFromPTXFile( ptx_path, "exception" ) );
  }
  m_context->setExceptionEnabled( RT_EXCEPTION_ALL, 1 );

  m_context->setRayGenerationProgram( ENTRY_COLOR_ONLY, m_context->createProgramFromPTXFile( ptx_path, "zp_color_only" ) ); 
  m_context->setRayGenerationProgram( ENTRY_GENERATE_SAMPLES, m_context->createProgramFromPTXFile( ptx_path, "zp_generate_samples" ) ); 
  m_context->setRayGenerationProgram( ENTRY_GATHER_SAMPLES, m_context->createProgramFromPTXFile( ptx_path, "zp_gather_samples" ) ); 
  m_context->setRayGenerationProgram( ENTRY_FLOAT_ATOMICS, m_context->createProgramFromPTXFile( ptx_path, "zp_scatter_samples" ) ); 
  m_context->setRayGenerationProgram( ENTRY_ZERO_SCATTER_BUFFERS, m_context->createProgramFromPTXFile( ptx_path, "zp_zero_scatter_buffers" ) ); 
  m_context->setRayGenerationProgram( ENTRY_SCATTER_DO_DIVIDE, m_context->createProgramFromPTXFile( ptx_path, "zp_scatter_do_divide" ) );
  m_context->setRayGenerationProgram( ENTRY_FIND_CONTRAST_LOCATIONS, m_context->createProgramFromPTXFile( ptx_path, "zp_find_contrast_locations" ) );
  m_context->setRayGenerationProgram( ENTRY_ADAPTIVE_RESAMPLE, m_context->createProgramFromPTXFile( ptx_path, "zp_adaptive_resample" ) );
}

void initOptiX( )
{
  // context 
  m_context = Context::create();
  m_context->setRayTypeCount( 1 );
  m_context->setEntryPointCount( NUM_ENTRY_POINTS );
  m_context->setStackSize( 2900 );
  m_context->setPrintEnabled( true );
  m_context->setPrintBufferSize( 4096 );
  // Because of zoneplate's use of atomics to an output buffer, we have to restrict the
  // number of devices to 1.
  std::vector<int> enabled_devices = m_context->getEnabledDevices();
  m_context->setDevices(enabled_devices.begin(), enabled_devices.begin()+1);

  createOptiXBuffers();

  setCameraProgram();
  updateOptiXWindowSize();

  setFilterType( filter_type );
  setRenderType( render_type );
  setFilterWidth( filter_width );
  setSQPP( sqrt_samples_per_pixel );
  setJitterAmount( jitter_amount );
  setGaussianAlpha( gaussian_alpha );
  setSincTau( sinc_tau );
  setCheckerboardRotate( checkerboard_rotate );
  setCheckerboardWidth( checkerboard_width );
  setAdaptiveContrastThreshold( adaptive_contrast_threshold );
  setContrastWindowWidth( contrast_window_width );

  // Prepare to run
  m_context->validate();
  m_context->compile();
}

void reshape( int w, int h ) {
  window_width = w;
  window_height = h;
  updateOptiXWindowSize();
  zoneplate_computed = false;
  glutPostRedisplay();
}

void trace( void )
{
  Buffer buffer = getOutputBuffer();
  RTsize buffer_width, buffer_height;
  buffer->getSize( buffer_width, buffer_height );

  m_context->launch( 0,
                   static_cast<unsigned int>(buffer_width),
                   static_cast<unsigned int>(buffer_height) );
}

void initGlut( int &argc, char **argv ) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);    
  glutInitWindowSize(window_width, window_height);
  glutCreateWindow("Zone Plate");
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutKeyboardFunc(keyboard);
  glutSpecialFunc(special);
#if !defined(__APPLE__)
  glewInit();
#endif

  sutilCurrentTime( &last_frame_time );
  frame_count = 0;
  last_frame_count = 0;

}

void printHelp(void)
{

    std::cerr <<
    "NVIDIA OptiX antialiasing demo\n"
    "\n"
    "Interactive options:\n"
    "  f                   Cycle through reconstruction filter types\n"
    "  w/W                 Decrease/increase filter width\n"
    "  s/S                 Decrease/increase samples per pixel\n"
    "  p                   Toggle rendering checkerboard / zoneplate images\n"
    "  c/C                 decrease/increase checkerboard size\n"
    "  a/A                 adjust contrast threshold for adaptive sampling\n"
    "  r                   toggle animation and FPS calculation (checkerboard only)\n"
    "  d                   Show adaptive sampling locations (adaptive sampling only)\n"
    "  m                   Cycle antialiasing methods\n"
    "  t                   Toggle text overlay\n"
    "  q or ESC            Quits the example\n";
}

int main( int argc, char** argv )
{
  try {

    printHelp();

    initGlut( argc, argv );
    initOptiX();

    glutMainLoop();
  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit( 1 );
  }
  return 0;
}

