#include <stdio.h>
#include <math.h>

#if defined (_CRAY) // this definition must be before nektar.h
#define zbesj_ ZBESJ
#endif
#include <veclib.h>

typedef struct {
  double re;
  double im;
} Cmplx;
#define SQRT2 1.41421356237309504880


// CC -g -n32 -o btest bessels_test.C -I../../include -L../../Hlib/IRIX64 -lvec -lftn -lm


main(){
  double alpha,mu,R,wnum;
  Cmplx jc1,jc2,c;
  int  KODE=1,NZ, IERR,N=1;
  double FNU=0.;
  
  wnum = 0.256;
  mu = 1.0/100;
  R = 0.5;
  
  alpha = R*sqrt(wnum/mu);
  jc2.re = -alpha/SQRT2;
  jc2.im = alpha/SQRT2;

  jc1.re = jc2.re;
  jc1.im = jc2.im;

  zbesj_(&jc1.re, &jc1.im, &FNU, &KODE, &N, &c.re, &c.im, &NZ, &IERR);
  
  printf("jc1.re: %lf jc1.im: %lf c.re: %lf, c.im: %lf\n",jc1.re,
	 jc1.im,c.re,c.im);
 
}

