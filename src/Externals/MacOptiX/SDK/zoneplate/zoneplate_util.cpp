
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
#include <string>

#include <optixu/optixu_math_stream.h>

float3 m_text_color           = make_float3( 0.95f, 0.f, 0.f );
float3 m_text_shadow_color    = make_float3( 0.10f );

void drawText( const std::string& text, float x, float y, void* font )
{
  // Save state
  glPushAttrib( GL_CURRENT_BIT | GL_ENABLE_BIT );

  glDisable( GL_TEXTURE_2D );
  glDisable( GL_LIGHTING );
  glDisable( GL_DEPTH_TEST);

  glColor3fv( &( m_text_shadow_color.x) ); // drop shadow
  // Shift shadow one pixel to the lower right.
  glWindowPos2f(x + 1.0f, y - 1.0f);
  for( std::string::const_iterator it = text.begin(); it != text.end(); ++it )
    glutBitmapCharacter( font, *it );

  glColor3fv( &( m_text_color.x) );        // main text
  glWindowPos2f(x, y);
  for( std::string::const_iterator it = text.begin(); it != text.end(); ++it )
    glutBitmapCharacter( font, *it );

  // Restore state
  glPopAttrib();
}

const char *getFilterName() {
  switch( filter_type ) {
    case FILTER_BOX: return "Box";
    case FILTER_TRIANGLE: return "Triangle";
    case FILTER_GAUSSIAN: return "Gaussian";
    case FILTER_MITCHELL: return "Mitchell(1/3, 1/3)";
    case FILTER_SINC: return "Sinc";
    default: return "UNKNOWN FILTER";
  }
}

const char *getAAName() {
  switch( aa_type ) {
    case AA_NONE: return "NONE";
    case AA_SAMPLE_GATHER: return "CUDA-based sample gather";
    case AA_FLOAT_ATOMICS: return "SM 2.0 floating-point atomics";
    case AA_CONTRAST_THRESHOLD: return "Adaptive contrast thresholds";
    default: return "UNKNOWN AA METHOD";
  }
}

const char *getImageName() {
  switch( render_type ) {
    case RENDER_ZONEPLATE: return "Zoneplate";
    case RENDER_CHECKERBOARD: return "Checkerboard";
    default: return "UNKNOWN RENDER TYPE";
  }
}

void setFilterType( zpFilterType ft ) {
  filter_type = ft;
  m_context["filter_type"]->setInt( filter_type );
  zoneplate_computed = false;
  glutPostRedisplay();
}

void setRenderType( zpRenderType rt )
{
  render_type = rt;
  m_context["render_type"]->setInt( render_type );
  zoneplate_computed = false;
  glutPostRedisplay();
}

void setAAType( zpAAType at )
{
  aa_type = at;
  zoneplate_computed = false;
  resizeOptiXBuffers();
  glutPostRedisplay();
}


void setFilterWidth( float fw ) {
  if (fw < .5) fw = .5;
  filter_width = fw;
  m_context["filter_width"]->setFloat( filter_width );
  zoneplate_computed = false;
  glutPostRedisplay();
}

void setCheckerboardRotate( float cr ) {
  while (cr < 0) cr += 360;
  while (cr >= 360) cr -= 360;
  checkerboard_rotate = cr;
  m_context["checkerboard_rotate"]->setFloat( checkerboard_rotate );
  zoneplate_computed = false;
  glutPostRedisplay();
}

void setCheckerboardWidth( int cw ) {
  if (cw < 1) cw = 1;
  checkerboard_width = cw;
  m_context["checkerboard_width"]->setInt( checkerboard_width );
  zoneplate_computed = false;
  glutPostRedisplay();
}

void setGaussianAlpha( float ga ) {
  gaussian_alpha = ga;
  m_context["gaussian_alpha"]->setFloat( gaussian_alpha );
  zoneplate_computed = false;
  glutPostRedisplay();
}

void setSincTau( float st ) {
  sinc_tau = st;
  m_context["sinc_tau"]->setFloat( sinc_tau );
  zoneplate_computed = false;
  glutPostRedisplay();
}

void setAdaptiveContrastThreshold( float act ) {
  if (act < 0) act = 0;
  if (act > 1) act = 1;
  adaptive_contrast_threshold = act;
  m_context["adaptive_contrast_threshold"]->setFloat( adaptive_contrast_threshold );
  zoneplate_computed = false;
  glutPostRedisplay();
}

void setContrastWindowWidth( int cww ) {
  if (cww < 1) cww = 1;
  contrast_window_width = cww;
  m_context["contrast_window_width"]->setUint( contrast_window_width );
  zoneplate_computed = false;
  glutPostRedisplay();
}

void setJitterAmount( float ja ) {
  if (ja < 0) ja = 0;
  if (ja > 1) ja = 1;
  jitter_amount = ja;
  m_context["jitter_amount"]->setFloat( jitter_amount );
  zoneplate_computed = false;
  glutPostRedisplay();
}

void setSQPP( int sqspp ) {
  static bool first = true;
  if (!first && sqspp == sqrt_samples_per_pixel) return;

  first = false;
  if (sqspp < 1) sqspp = 1;
  sqrt_samples_per_pixel = sqspp;
  m_context["sqrt_samples_per_pixel"]->setUint( sqrt_samples_per_pixel );
  resizeOptiXBuffers();
  zoneplate_computed = false;
  glutPostRedisplay();
}
