#ifndef H_NEKTAR
#define H_NEKTAR

/*---------------------------------------------------------------------------*
 *                        RCS Information                                    *
 *                                                                           *
 * $Source: /homedir/cvs/Nektar/include/nektar.h,v $  
 * $Revision: 1.21 $
 * $Date: 2009-11-24 19:05:33 $    
 * $Author: bscarmo $  
 * $State: Exp $   
 *---------------------------------------------------------------------------*/

#include <stdlib.h>
#ifdef PARALLEL
#include <mpi.h>
#define exit(a) {MPI_Abort(MPI_COMM_WORLD, -1); }
extern "C"
{
#include "gs.h"
}
#endif

#include <hotel.h>
#include <stokes_solve.h>
using namespace MatSolve;

#define MAXDIM 3

/* general include files */
#include <math.h>
#include <veclib.h>

/* parameters */
#define HP_MAX  128   /* Maximum number of history points */

/* SLVTYPE Flags */
typedef enum {
    Splitting,			/* Splitting scheme                        	*/
    StokesSlv,          /* Stokes solver                           	*/
    SubStep,            /* DG-Advection substepping                	*/
    SubStep_StokesSlv,  /* DG-Stokes substepping                   	*/
    Adjoint_StokesSlv,	/* Adjoint Stokes							*/
    TurbCurr,           /* Splitting scheme with turbulent current  */
	SubIt_Bmotion,		/* Sub-Iteration scheme for fluid-structure */
	SteadyNewton		/* Steady Navier-Stokes via Newton's method */
} SLVTYPE;

typedef enum {                  /* ......... ACTION Flags .......... */
    Rotational,                   /* N(U) = U x curl U                 */
    Convective,                   /* N(U) = U . grad U                 */
    Stokes,                       /* N(U) = 0  [drive force only]      */
    Alternating,                  /* N(U^n,T) = U.grad T               */
    /* N(U^(n+1),T) = div (U T)          */
    Linearised,                   /* Linearised convective term        */
    StokesS,                      /* Steady Stokes Solve               */
    Oseen,                        /* Oseen flow                        */
    OseenForce,
    StokesForce,                  /* Stokes forcing terms              */
    LambForce,                    /* N(U) = u^2 Curl w - j             */
    Pressure,                     /* div (U'/dt)                       */
    Viscous,                      /* (U' - dt grad P) Re / dt          */
    Prep,                         /* Run the PREP phase of MakeF()     */
    Post,                         /* Run the POST phase of MakeF()     */
    Transformed,                  /* Transformed  space                */
    Physical,                     /* Phyiscal     space                */
    TransVert,                    /* Take history points from vertices */
    TransEdge,                    /* Take history points from mid edge */
    Poisson   = 0,
    Helmholtz = 1,
    Laplace   = 2
} ACTION;

typedef struct hpnt
{           /* ......... HISTORY POINT ......... */
	int         id         ;      /* ID number of the element          */
	int         i, j, k    ;      /* Location in the mesh ... (i,j,k)  */
	char        flags             /* The fields to echo.               */
	[_MAX_FIELDS];      /*   (up to maxfields)               */
	ACTION      mode       ;      /* Physical or Fourier space         */
	struct hpnt *next      ;      /* Pointer to the next point         */
}
HisPoint;

typedef enum {
	BCGStab,
	CGS
} ITERSLVTYPE;

// Ce107
typedef struct intepts
{
	int     npts;
	Coord    X;
	double **ui;
}
Intepts;


typedef struct gf_
{                  /* ....... Green's Function ........ */
	int           order; 					// Time-order of the current field
	Bndry        *Gbc; 						// Special boundary condition array
	Element_List *basis; 					// Basis velocity (U or W)
	Element_List *Gv[MAXDIM][_MAX_ORDER]; 	// Green's function velocities
	Element_List *Gp        [_MAX_ORDER];	// Green's function pressure
	double        Fg        [_MAX_ORDER]; 	// Green's function "force"
}
GreensF;


/* local structure for global number */
typedef struct gmapping
{
	int nvs;
	int nvg;
	int nes;
	int neg;
	int *vgid;
	int *egid;
	int nfs;
	int nfg;
	int *fgid;

	int nsolve;
	int nglobal;
	int nv_psolve;
}
Gmap;

/* time dependent function inputs -- ie. womersley flow */
typedef struct time_str
{  /* string input      */
	char *TimeDepend;
}
Time_str;

/* info for a function of the form:
   t = (ccos[i]*cos(wnum[i]*t) + csin[i]*sin(wnum[i]*i*t)) */

typedef struct time_fexp
{  /* fourier expansion */
	int nmodes;
	double wnum;
	double *ccos;
	double *csin;
}
Time_fexp;

typedef enum{
    ENCODED,                /* fourier encoded values */
    RAW,                    /* actual measure,ents    */
    PRESSURE,
    MASSFLOW
} TwomType;

typedef struct time_wexp
{  /* Womersley expansion fourier expansion */
	int nmodes;
	int nraw;
	double scal0;
	double radius;
	double axispt[MAXDIM];
	double axisnm[MAXDIM];
	double wnum;
	double *ccos;
	double *csin;
	double *raw;
	TwomType type;
	TwomType form;
}
Time_wexp;

typedef union tfuninfo {
	Time_str  string [MAXDIM];
	Time_fexp fexpand[MAXDIM];
	Time_wexp wexpand;
} TfunInfo;

typedef enum{
    String,                 /* input string            */
    Fourier,                /* input fourier expansion */
    Womersley               /* input womersley expansion */
} TfunType;

typedef struct tfuct
{
	TfunType    type;             /* type of time function input */
	TfunInfo    info;             /* info for function           */
}
Tfunct;


#ifdef BMOTION2D
typedef enum motiontype {
    Prescribed,             // no body motion
    Heave,                 // heave motion only
    InLine,                // Inline motion only
    Rotation,              // Rotation only
    HeaveRotation          // heave and rotation
} MotionType;

typedef struct motion2d
{
	MotionType motiontype; /* enumeration list with motion type           */
	double stheta;       /* inertial moment of the section               */
	double Itheta;       /* static moment of the section                 */
	double mass;         /* mass of the section                          */
	double cc[2];        /* damping coefficients for each degree of      */
	/* freedom, cc[0] for y and cc[1] for theta     */
	double ck[2];        /* stiffness coefficients                       */
	double cl[2];        /* scaling constants for force and moment       */
	double dp[2];        /* displacement at previous time level          */
	double dc[2];        /* displacement at current time level           */
	double vp[2];        /* first derivatives at previous time level     */
	double vc[2];        /* first derivatives at current time level      */
	double ap[2];        /* second derivatives at previous time level    */
	double ac[2];        /* second derivatives at current time level     */

	double fp[3];        /* for storing forces						   */
	double fpp[3];        /* for storing forces						   */
	double fc[3];		 /* Forces at current time. Used for substep scheme. */
	double aot;          /* Angle of attack in HM mode                   */
	double x[3];
	double alpha[3];
	double Ured_y;
	double Ured_alpha;
	double zeta_y;
	double zeta_alpha;
	double mass_ratio;
	double inertial_mom;
}
Motion2d;
#endif

#ifdef ALE

// Data structure for elastic base
typedef struct
{
	int tag;			// Elastic base tag

	double mx, my,		// Mass coefficients
		   cx, cy,		// Damping coefficients
		   kx, ky;		// Spring stiffness coefficients

	double dxi, dyi,	// Initial displacements
		   vxi, vyi;	// Initial velocities

	double ax, ay,		// Acceleration
		   vx, vy,		// Velocity
		   dx, dy;		// Displacement

	double axp, ayp,		// Acceleration - previous timestep
		   vxp[3], vyp[3],	// Velocity - previous timesteps
		   dxp[3], dyp[3];	// Displacement - previous timesteps

	double auxvx, 		// Auxiliar velocity variables to make the different
		   auxvy;		// structural and flow integration schemes consistent.

	double fx, fy;			// Forces
	double fxp[3], fyp[3];	// Forces - previous timesteps

}
ElastBase;

// Data structure for spring mesh
typedef struct
{
	double x, y;		// Node coordinates
	double *xp, *yp;	// Node coordinates - previous timesteps.
	double *k;			// Link stiffness array

	double dx,		// x displacements
	dy,				// y displacements
	pdx,			// x displacement in previous time step
	pdy,			// y displacement in previous time step
	adx,			// auxiliar variable for x displacement
	ady;			// auxiliar variable for y displacement

	int nneigh;			// Number of neighbour nodes
	int *neigh;			// Neighbour nodes array

	bool defx, defy;	// True if the node is allowed to move in the x and y
	// directions respectively

	int nel;			// Number of elements that has this node as a vertex
	int *elid,			// Elements id array
	*vert;			// Vertex number in each element - array
}
SpringNode;

#endif


/* Solution Domain contains global information about solution domain */
typedef struct dom
{
	int      step;
	char     *name;           /* Name of run                       */
	char     **soln;
	double   dt;

	FILE     *fld_file;       /* field file pointer                */
	FILE     *dat_file;       /* field file pointer                */
	FILE     *his_file;       /* history file pointer              */
	FILE     *fce_file;       /* force file                        */
	FILE     *inflow_file;    /* inflow_file                       */ //AnaPlata
#ifdef FLOWRATE

	FILE     *flo_file;       /* flow rate file                    */
#endif

	HisPoint *his_list;       /* link list for history points      */
	Tfunct   *Tfun;           /* time dependent into conditions    */

	Element_List  *U, *V, *W, *P;  /* Velocity and Pressure fields      */
	Element_List  *Uf;             /* --------------------------------- */
	Element_List  *Vf;             /*        Multi-step storage         */
	Element_List  *Wf;             /* --------------------------------- */
	Element_List  *Pf;
	Element_List  *Lfx,*Lfy;       /* forcing terms to lamb vector      */

	MatSolve::StokesMatrix *StkSlv;

	double **u;                   /* Field storage - physical space */
	double **v;                   /* Field storage - physical space */
	double **w;                   /* Field storage - physical space */

	double **uf;                   /* Non-linear forcing storage */
	double **vf;                   /* Non-linear forcing storage */
	double **wf;                   /* Non-linear forcing storage */

	double **lfx;                 /* lamb vector storage */
	double **lfy;


	double **us;                  /* multistep storage - transformed space */
	double **vs;
	double **ws;
	double **ps;

	double **ul;                  /* Lagrangian velocity for Op Int Spl. */
	double **vl;
	double **wl;

	Bndry    *Ubc,*Vbc,*Wbc;      /* Boundary  conditons               */
	Bndry    *Pbc;
#ifdef ALI

	Bndry    *dUdt, *dVdt;
#endif

	Bsystem  *Usys;               /* Velocity Helmholtz matrix system  */
	Bsystem  *Vsys;               /* Velocity Helmholtz matrix system  */
	Bsystem  *Wsys;               /* Velocity Helmholtz matrix system  */
	Bsystem  *Pressure_sys;       /* pressure Poisson   matrix system  */

	double   **ForceFuncs;        /* storage for variable forcing      */
	char     **ForceStrings;      /* string definition of forcing      */

	//  Metric     *kinvis;

	// ALE structures
#ifdef ALE

	Element_List **MeshX;		// Mesh coordinates for ALE formulation
	Element_List **MeshV;		// Mesh velocities
	Element_List *MeshVf;		// RHS mesh velocities system
	Bndry        **MeshBCs;		// Boundary conditions for mesh movement
	Bsystem      **Mesh_sys;	// Matrix systems for mesh velocities
	double *mu, *mv, *mw;		// Mesh velocities modes in previous timestep
	double **mx, **my, **mz;	// Mesh coordinates modes in previous timesteps
	double *au, *av;			// Advection velocity (flow veloc - mesh veloc)
	// (physical space)
	ElastBase *ElastB;			// Elastic Base data pointer
	int neb;					// Number of elastic bases

	int nsnodes;				// Number of nodes in the spring mesh
	int nsn_solve;				// Number of nodes to solve for
	SpringNode *snode;			// Array of nodes in the spring mesh

#ifdef SMOOTHER
	Element_List *Psegs;		// Data structure for smoothing capability
	Element_List *PsegsF;
	Bsystem      *Psegs_sys;

	Element_List *Msegs;
	Element_List *MsegsUF;
	Element_List *MsegsVF;
	Bsystem      *Msegs_sys;
#endif

#endif

#ifdef BMOTION2D

	FILE     *bdd_file, *bda_file;       /* motion of body file              */
	FILE     *bgy_file;                  /* energy data                      */

	Motion2d   *motion2d;                /* Body motion info                 */
#endif

#ifdef MAP
	// Ce107
	FILE     *int_file;       /* interpolation file pointer        */
	Map      *mapx,  *mapy;   /* Mapping in x and y                */
	MStatStr *mstat;          /* Moving Statistics information     */
	Intepts  *int_list;       /* link list for interpolation points*/
#endif
}
Domain;

/* function in drive.c  */
void MakeF   (Domain *omega, ACTION act, SLVTYPE slvtype);
void solve(Element_List *U, Element_List *Uf,Bndry *Ubc,Bsystem *Ubsys,
		   SolveType Stype,int step);
int Navier_Stokes(Domain *Omega, double t0, double tN);
void Reset(Domain *Omega);
void InvMult(Element_List *U, Bsystem *B);

/* functions in prepost.c */
void      parse_args (int argc,  char **argv);
void      PostProcess(Domain *omega, int, double);
Domain   *PreProcess (int argc, char **argv);
void      set_vertex_links(Element_List *UL);
void LocalNumScheme  (Element_List *E, Bsystem *Bsys, Gmap *gmap);
Gmap *GlobalNumScheme(Element_List *E, Bndry *Ebc);
void free_gmap(Gmap *gmap);
void free_Global_info(Element_List *Mesh, Bndry *Meshbc,
                      Gmap *gmap, int lnel);
Bsystem *gen_bsystem(Element_List *UL, Bndry *Ebc);
//gen_bsystem function for utilities
Bsystem *gen_bsystem(Element_List *UL, Gmap *gmap);

/* functions in io.c */
void      ReadParams     (FILE *rea);
void      ReadPscals     (FILE *rea);
void      ReadLogics     (FILE *rea);
Element_List  *ReadMesh       (FILE *rea,char *);
void      ReadKinvis     (Domain *);
void      ReadICs        (FILE *, Domain *);
void      ReadDF         (FILE *fp, int nforces, ...);
void      summary        (void);
void      ReadSetLink    (FILE *fp, Element_List *U);
void      ReadSetLink    (FILE *fp, Element *U);
Bndry    *ReadBCs        (FILE *fp, Element *U, AdvVel *vel = (AdvVel *) NULL); //AnaPlata
Bndry *ReadMeshBCs (FILE *fp, Element_List *Mesh);
Bndry    *bsort          (Bndry *, int );
void      read_connect   (FILE *name, Element_List *);
void      ReadOrderFile  (char *name,Element_List *E);
void      ReadHisData    (FILE *fp, Domain *omega);
void      ReadSoln       (FILE* fp, Domain* omega);
void      ReadDFunc      (FILE *fp, Domain *Omega);
void      ReadWave       (FILE *fp, double **wave, Element_List *U);
void      ReadTimeDepend (FILE *fp, Domain *omega);
void	  ReadInflow     (FILE *fp, Domain *omega, Element_List *Mesh); //AnaPlata
void	  ReadInflow2D   (FILE *fp, Domain *omega, Bndry *Meshbc);        //AnaPlata          

/* structure specific to bwoptim and recurSC */
typedef struct facet
{
	int  id;
	struct facet *next;
}
Facet;

typedef struct fctcon
{
	int ncon;
	Facet *f;
}
Fctcon;


/* function in bwoptim.c */
void bandwidthopt (Element *E, Bsystem *Bsys, char trip);
void MinOrdering   (int nsols,  Fctcon *ptcon, int *newmap);
void addfct(Fctcon *con, int *pts, int n);
void free_Fctcon   (Fctcon *con, int n);

/* functions in recurrSC.c */
void Recursive_SC_decom(Element *E, Bsystem *B);

/* functions in convective.c */
void VdgradV (Domain *omega);
void CdgradV (Domain *omega);

/* functions in DN.C */
void DN(Domain *omega);
void SetBase(Domain *omega);

/* functions in rotational.c */
void VxOmega (Domain *omega);

/* functions in divergenceVv.c */
void DivVv (Domain *omega);

/* functions in lambforce.C */
void LambForcing(Domain *omega);

/* functions in pressure.c */
Bndry *BuildPBCs (Element_List *P, Bndry *temp);
void   SetPBCs   (Domain *omega);
void Set_Global_Pressure(Element_List *Mesh, Bndry *Meshbcs);
void Replace_Numbering(Element_List *UL, Element_List *GUL);
void Replace_Local_Numbering(Element_List *UL, Element_List *GUL);
void set_delta(int Je);

/* function in stokes.c      */
void StokesBC (Domain *omega);

/* functions in analyser.c   */
void Analyser (Domain *omega, int step, double time);

/* functions in forces */
void  forces (Domain *omega, int step, double time); 	//3d version
void  forces (Domain *omega, double time);				//2d version
void  Forces (Domain *omega, double time, double *F,int writoutput);
void surf_inflow(Domain *omega, int step, double time); //AnaPlata

/* functions in sections */
int cnt_srf_elements(Domain *omega);
Element_List *setup_surflist(Domain *omega, Bndry *Ubc, char type);

#ifdef ALE
// Functions in ALE.C
void Update_Mesh(Domain *omega, int Je, double dt);
void Integrate_xy(int Je, double dt, Element_List *mX, Element_List *mU,
                  double **mx);
void Set_Mesh(Domain *omega);
void Update_Bndry_Geofac(Bndry *Ubc);
void M_Solve(Element_List *U, Element_List *Uf, Bndry *Ubc, Bsystem *Ubsys,
             SolveType Stype);
void Setup_ALE   (Domain *Omega);
void Update_Mesh_Velocity(Domain *Omega);
void Set_ALE_ICs(Domain *omega);
void Setup_Elast_Base(Domain *omega);
void Integrate_Elast_Base(Domain *omega, double time, double dt, int step);
void Update_Bsystem(Element_List *U, Bndry *Ubc, Bsystem *Ubsys, double *us,
                    SolveType Stype);
bool Check_Conv_ALE(Domain *omega, double time, int step, double dt);

// Functions in Matrix.C
void Selec_Bsystem_mem_free(Bsystem *Ubsys, Element_List *U);
void Selec_GenMat(Element_List *U, Bndry *Ubc, Bsystem *Ubsys, Metric *lambda,
                  SolveType Stype, int zel);

// old ALE functions
void set_soliton_ICs(Element_List *U, Element_List *V);
void set_soliton_BCs(Bndry *Ubc, Bndry *Vbc, char ch);
void update_paddle(Element_List *, Bndry *);
#endif

// Functions in smoother
Element_List *setup_seglist(Bndry *Ubc, char type);
Bsystem      *setup_segbsystem(Element_List *seg_list);
void          fill_seglist(Element_List *seg_list, Bndry *Ubc);
void          fill_bcs(Element_List *seg_list, Bndry *Ubc);
void          update_seg_vertices(Element_List *, Bndry *);
void update_surface(Element_List *EL, Element_List *EL_F, Bsystem *Bsys,
				    Bndry *BCs);
//void smooth_surface(Element_List *EL, Element_List *EL_F,
//		    Bsystem *Bsys, Bndry *BCs);
void smooth_surface(Domain *, Element_List *segs, Bsystem *Bsys, Bndry *BCs);
void test_surface(Bndry *Ubc, char type);

// functions in magnetic
void Set_Global_Magnetic(Element_List *Mesh, Bndry *Meshbcs);
void ReadAppliedMag(FILE* fp, Domain* omega);

double cfl_checker     (Domain *omega, double dt);
double full_cfl_checker(Domain *omega, double dt, int *eid_max);

// functions in mpinektar.C
void init_comm(int*, char ***);
void exit_comm();
void exchange_sides(int Nfields, Element_List **Us);
void SendRecvRep(void *buf, int len, int proc);

/* functions in mlevel.C */
void Mlevel_SC_decom(Element_List *E, Bsystem *B);

/* functions in womersley.C */
void WomInit(Domain *Omega);
void SetWomSol (Domain *Omega, double time, int save);
void SaveWomSol(Domain *Omega);

void zbesj_(double *ZR, double *ZI, double *FNU, int *KODE, int *N,
            double *CYR, double *CYI, int *NZ, int *IERR);
void WomError    (Domain *omega, double time);
void SetWomField (Domain *omega, double *u, double *v, double *w, double time);

/* functions in wannier.C */
#ifdef HEIMINEZ
void Heim_define_ICs (Element_List  *V);
void Heim_reset_ICs  (Element_List **V);
void Heim_error      (Element_List **V);
#endif
/* functions in structsolve.C */
void CalcMeshVels(Domain *omega);


#ifdef MAP
typedef struct mppng
{            /* ......... Mapping ............... */
	int       NZ                  ; /* Number of z-planes                */
	double    time                ; /* Current time                      */
	double   *d                   ; /* Displacement                      */
	double   *z                   ; /*   (z-deriv)                       */
	double   *zz                  ; /*   (zz-deriv)                      */
	double   *t                   ; /* Velocity                          */
	double   *tt                  ; /* Acceleration                      */
	double   *tz                  ; /*   (tz-deriv)                      */
	double   *tzz                 ; /*   (tzz-deriv)                     */
	double   *f                   ; /* Force                             */
}
Map;

typedef struct mstatstr
{         /* Moving Statistics information     */
	double *x;                      /* vector holding the x-coordinate   */
	double *y;                      /* vector holding the y-coordinate   */
	int    *sgnx;                   /* vector holding the sign of dx/dt  */
	int    *sgny;                   /* vector holding the sign of dy/dt  */
	int    *nstat;                  /* vector holding the # of samples   */
	int     n;                      /* number of sampling (x,y) points   */
}
MStatStr;
#endif

/* ce107 changes begin */
int       backup         (char *path1);
void      ReadIntData    (FILE *fp, Domain *omega);
void      ReadSoln       (FILE *fp, Domain *omega);
void      ReadMStatPoints(FILE* fp, Domain* omega);
void      averagefields  (Domain *omega, int nsteps, double time);
double    VolInt         (Element_List *U, double shift);
double    VolInt         (Element_List *U);

double            L2norm (Element_List *V);
void       average_u2_avg(Domain *omega, int step, double time);
void             init_avg(Domain *omega);

/* functions in interp */
void interp(Domain *omega);

/* functions in nektar.C */
void AssembleLocal(Element_List *U, Bsystem *B);

/* functions in dgalerkin.c */
void set_elmt_sides    (Element_List *E);
void set_sides_geofac  (Element_List *EL);
void Jtransbwd_Orth    (Element_List *EL, Element_List *ELf);
void Jtransbwd_Orth_hj (Element_List *EL, Element_List *ELf);
void EJtransbwd_Orth   (Element *U, double *in, double *out);
void InnerProduct_Orth (Element_List *EL, Element_List *ELf);
int  *Tri_nmap         (int l, int con);
int  *Quad_nmap        (int l, int con);

/* subcycle.C */
void SubCycle(Domain *Omega, int Torder);
void Upwind_edge_advection(Element_List *U, Element_List *V,
                           Element_List *UF, Element_List *VF);
void Add_surface_contrib(Element_List *Us, Element_List *Uf);
void Fill_edges(Element_List *U, Element_List *Uf, Bndry *Ubc,double t);


#ifdef BMOTION2D
/* functions in bodymotion.c */
void set_motion (Domain *omega, char *bddfile, char *bdafile);
void Bm2daddfce (Domain *omega);
void ResetBc    (Domain *omega);
void IntegrateStruct(Domain *omega, double time, int step, double dt);
bool checkconv(Domain *omega, double time, int step, double dt);
#endif

// MSB: Added for PSE------------------------------------
// Needs to be included after Domain is defined        //
#include "stokes_solve_F.h"                              //

#endif
