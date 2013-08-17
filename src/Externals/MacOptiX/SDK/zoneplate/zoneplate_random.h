#ifndef ZONEPLATE_RANDOM_H
#define ZONEPLATE_RANDOM_H

extern float genrand_real2();

inline float RandomFloat() {
  return genrand_real2();
}

float get_random_float( unsigned int & m_w, unsigned int & m_z );

#endif /* ZONEPLATE_RANDOM_H */

