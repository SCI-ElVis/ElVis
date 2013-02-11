/* Header file for the Stochastic Nektar *
   by Dongbin Xiu
*/
#define KPMAX 100

//void Setup_spBC( int num, sp_Bndry* spBC, int nRAND, int nP, double ***e );
Domain *s_PreProcess(int argc, char **argv, double ***e);
void s_Convect( Domain *Omega, int KP, int ***e );
void s_Integrate( double dt, Element_List *U, Element_List *Uf );
//int*** Setup_e(int nRAND, int nP, int KP);
double*** Setup_e(int nRAND, int nP, int KP);
double*** Setup_e();
void s_Analyser (Domain *omega, int step, double time, int KP);
void s_PostProcess(Domain *omega, int step, double time, int KP);
void sPostProcess(Domain *omega, int step, double time, double ***e);
void s_SetPBCs(Domain *omega, int k);

void s_Convect( Domain *Omega, int KP, double ***e );
void s_Convect( Domain *Omega);
void s_ReadSoln(FILE* fp, Domain *omega, int KP);

void s_Read_SKinvis (FILE *fp, Domain *omega);
void s_Read_RHS (FILE *fp, Domain *omega);
void s_Read_SCapacity (FILE *fp, Domain *omega);
//void s_Constant_Capacity(FILE *fp, Domain *omega);
/* Functions for General Askey-Chaos */
int Chaos_Terms( int N, int p );
double gammaF(double x);
double Factorial(int N);
double Binomial(int m, int n);
double Pochhammer(int a, int n);

/* Functions for Jacobi-Chaos */
void   Jacobi_Triplets(int N, int p, int P, double alpha, double beta, double ***e);
void   Jacobi_Triplets_1D(int N, int p, int P, double alpha, double beta, double ***e);
void   Jacobi_Triplets_2D(int N, int p, int P, double alpha, double beta, double ***e);
void   Jacobi_Triplets_3D(int N, int p, int P, double alpha, double beta, double ***e);
void   Jacobi_Triplets_4D(int N, int p, int P, double alpha, double beta, double ***e);

/* Functions for Laguerre-Chaos */
void   Laguerre_Triplets(int N, int p, int P, double alpha, double ***e);
void   Laguerre_Triplets_1D(int N, int p, int P, double alpha, double ***e);
void   Laguerre_Triplets_2D(int N, int p, int P, double alpha, double ***e);
void   Laguerre_Triplets_3D(int N, int p, int P, double alpha, double ***e);
void   Laguerre_Triplets_4D(int N, int p, int P, double alpha, double ***e);

double LaguerrePoly(int degree, double alpha, double x);
double LaguerrePolyDerivative(int degree, double alpha, double x);
void   LaguerreZeros(int degree, double alpha, double *z);
void   LaguerreZW(int degree, double alpha, double * z, double *w);
void   LaguerreF(int np, double *x, double *f, int degree, double alpha);


/* Functions for Hermite-Chaos */
double HermitePoly(int degree, double x);
double HermitePolyDerivative(int degree, double x);
void   HermiteZeros(int degree, double *z);
void   HermiteZW(int degree, double *z, double *w);
void   HermiteF(int np, double *x, double *f, int degree);

void   Hermite_Triplets(int N, int p, int P, double ***e);
void   Hermite_Triplets_1D(int N, int p, int P, double ***e);
void   Hermite_Triplets_2D(int N, int p, int P, double ***e);
void   Hermite_Triplets_3D(int N, int p, int P, double ***e);
void   Hermite_Triplets_4D(int N, int p, int P, double ***e);

/* Functions for Charlier-Chaos */
void   Charlier_Triplets(int N, int p, int P, double alpha, double ***e);
void   Charlier_Triplets_1D(int N, int p, int P, double a, double ***e);
void   Charlier_Triplets_2D(int N, int p, int P, double a, double ***e);
void   Charlier_Triplets_3D(int N, int p, int P, double a, double ***e);
void   Charlier_Triplets_4D(int N, int p, int P, double a, double ***e);

void CharlierF(int np, double a, double *f, int degree);

/* Functions for Krawtchouk-Chaos */
void   Krawtchouk_Triplets(int N,int p,int P, int M,double pp, double ***e);
void   Krawtchouk_Triplets_1D(int N,int p,int P, int M,double pp, double ***e);
void   Krawtchouk_Triplets_2D(int N,int p,int P, int M,double pp, double ***e);
void   Krawtchouk_Triplets_3D(int N,int p,int P, int M,double pp, double ***e);
void   Krawtchouk_Triplets_4D(int N,int p,int P, int M,double pp, double ***e);

void KrawtchoukF(int N, double p, double *f, int degree);
void KrawtchoukZW(int N, double p, double *w);
double KrawtchoukPoly(int N, double p, int degree, int x);

/* Functions for Charlier-Chaos */
void   Charlier_Triplets(int N,int p,int P, double lambda, double ***e);
void   Charlier_Triplets_1D(int N,int p,int P, double lambda, double ***e);
void   Charlier_Triplets_2D(int N,int p,int P, double lambda, double ***e);
void   Charlier_Triplets_3D(int N,int p,int P, double lambda, double ***e);
void   Charlier_Triplets_4D(int N,int p,int P, double lambda, double ***e);

void CharlierF(double lambda, int N, double *f, int degree);
void CharlierZW(double lambda, int N, double *w);
double CharlierPoly(double lambda, int degree, int x);

