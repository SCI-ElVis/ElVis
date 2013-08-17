#ifndef ZONEPLATE_UTIL_H
#define ZONEPLATE_UTIL_H

#include <string>

#include <optixu/optixpp_namespace.h>
#include "zoneplate_common.h"

using namespace optix;

typedef enum { 
  AA_NONE = 0,
  AA_SAMPLE_GATHER,
  AA_FLOAT_ATOMICS,
  AA_CONTRAST_THRESHOLD,
  NUM_AA_TYPES
} zpAAType;

extern zpAAType aa_type;
extern zpFilterType filter_type;
extern zpRenderType render_type;

extern unsigned int window_width, window_height;
extern Context m_context;
extern bool zoneplate_computed;
extern int sqrt_samples_per_pixel;
extern float filter_width;
extern float gaussian_alpha;
extern float sinc_tau;
extern float checkerboard_rotate;
extern int checkerboard_width;
extern int contrast_window_width;
extern float adaptive_contrast_threshold;
extern float jitter_amount;
extern bool text_overlay;

void drawText( const std::string& text, float x, float y, void* font );

const char *getFilterName();
const char *getAAName();
const char *getImageName();

void setFilterType( zpFilterType ft );
void setRenderType( zpRenderType rt );
void setAAType( zpAAType at );
void setFilterWidth( float fw );
void setCheckerboardRotate( float cr );
void setCheckerboardWidth( int cw );
void setGaussianAlpha( float ga );
void setSincTau( float st );
void setSQPP( int sqspp );
void setAdaptiveContrastThreshold( float act );
void setContrastWindowWidth( int cww );
void setJitterAmount( float ja );
void createOptiXBuffers();
void resizeOptiXBuffers();

#endif /* ZONEPLATE_UTIL_H */

