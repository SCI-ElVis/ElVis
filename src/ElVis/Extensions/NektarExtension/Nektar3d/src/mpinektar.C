/*
 * mpiprism -- MPI-F version of prism3d
 *
 * C.H. Crawford -- cait@cfm.brown.EDU
 * (from Dave Newman's SGI original pvm3 version)
 *
 * ------------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <math.h>
#include <veclib.h>
#include "nektar.h"

#ifdef PARALLEL
#include "gs.h"
#include "mpi.h"


void unreduce (double *x, int n);
void reduce   (double *x, int n, double *work);
static int numproc;
static int my_node;

void init_comm (int *argc, char *argv[])
{
  int info, nprocs,                      /* Number of processors */
      mytid;                             /* My task id */

  info = MPI_Init (argc, &argv);                 /* Initialize */
  if (info != MPI_SUCCESS) {
    fprintf (stderr, "MPI initialization error\n");
    exit(1);
  }
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);         /* Number of processors */
  MPI_Comm_rank(MPI_COMM_WORLD, &mytid);          /* my process id */

  numproc = nprocs;
  my_node = mytid;

  MPI_Barrier  (MPI_COMM_WORLD);                  /* sync before work */

  return;
}


void exi_comm(){

  MPI_Finalize();

  return;
}



void gsync ()
{
  int info;

  info = MPI_Barrier(MPI_COMM_WORLD);

  return;
}


int numnodes ()
{
  int np;

  MPI_Comm_size(MPI_COMM_WORLD, &np);         /* Number of processors */

  return np;
}


int mynode ()
{
  int myid;

  MPI_Comm_rank(MPI_COMM_WORLD, &myid);          /* my process id */
  
  return myid;
}

 
void csend (int type, void *buf, int len, int node, int pid)
{

  MPI_Send (buf, len, MPI_BYTE, node, type, MPI_COMM_WORLD);
  
  return;
}

void crecv (int typesel, void *buf, int len)
{
  MPI_Status status;

  MPI_Recv (buf, len, MPI_BYTE, MPI_ANY_SOURCE, typesel, MPI_COMM_WORLD, &status);
  
  return;
}


void msgwait (MPI_Request *request)
{
  MPI_Status status;

  MPI_Wait (request, &status);

  return;
}

double dclock(void)
{
  double time;

  time = MPI_Wtime();

  return time;

}
void gimax (int *x, int n, int *work) { 
  register int i;

  MPI_Allreduce (x, work, n, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  /* *x = *work; */
  icopy(n,work,1,x,1);

  return;
}

void gdmax (double *x, int n, double *work)
{
  register int i;

  MPI_Allreduce (x, work, n, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  /* *x = *work; */
  dcopy(n,work,1,x,1);

  return;
}

void gdsum (double *x, int n, double *work)
{
  register int i;

  MPI_Allreduce (x, work, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  /* *x = *work; */
  dcopy(n,work,1,x,1);

  return;
}

void gisum (int *x, int n, int *work)
{
  register int i;

  MPI_Allreduce (x, work, n, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  /* *x = *work; */
  icopy(n,work,1,x,1);

  return;
}

void ifexists(double *in, double *inout, int *n, MPI_Datatype *size);

void BCreduce(double *bc, Bsystem *Ubsys){
#if 1
  gs_gop(Ubsys->pll->known,bc,"A");
#else
  register int i;
  int      ngk = Ubsys->pll->nglobal - Ubsys->pll->nsolve;
  int      nlk = Ubsys->nglobal - Ubsys->nsolve;
  double *n1,*n2;
  static MPI_Op MPI_Ifexists = NULL;
  
  if(!MPI_Ifexists){
    MPI_Op_create((MPI_User_function *)ifexists, 1, &MPI_Ifexists);
  } 

  n1 = dvector(0,ngk-1);
  n2 = dvector(0,ngk-1);

  memset(n1,'\0',ngk*sizeof(double));
  memset(n2,'\0',ngk*sizeof(double));

  /* fill n1 with values from bc  */
  for(i = 0; i < nlk; ++i) n1[Ubsys->pll->knownmap[i]] = bc[i];
  
  /* receive list from other processors and check against local */
  MPI_Allreduce (n1, n2, ngk, MPI_DOUBLE, MPI_Ifexists, MPI_COMM_WORLD);
  /* fill bc with values values from  n1 */
  for(i = 0; i < nlk; ++i) bc[i] = n2[Ubsys->pll->knownmap[i]];

  free(n1);  free(n2);
#endif
}

void GatherBlockMatrices(Element *U,Bsystem *B){
  register int i,j;
  int nes = B->ne_solve;
  int nfs = B->nf_solve;
  int nel = B->nel;
  int l, *map, start, one=1;
  double *edge, *face;
  Edge   *e;
  Face   *f;
  extern Element *Mesh;
  struct gather_scatter_id *gather;

  switch(B->Precon){
  case Pre_Block:
    edge = B->Pmat->info.block.iedge[0];
    face = B->Pmat->info.block.iface[0];
    break;
  case Pre_LEnergy:
    edge = B->Pmat->info.lenergy.iedge[0];
    face = B->Pmat->info.lenergy.iface[0];
    break;
  default:
    error_msg(Unknown preconditioner in GatherBlockMatrices);
    break;
  }


  /* make up numbering list based upon solvemap */
  /* assumed fixed L order */
  l = (LGmax-2);
  l = l*(l+1)/2;
  map = ivector(0,l*nes);
  
  for(i = 0; i < nel; ++i)
    for(j = 0; j < Nedge; ++j){
      e = U[i].edge + j;
      if(e->gid < nes){
	start = Mesh[pllinfo.eloop[e->eid]].edge[j].gid*l;
	iramp(l,&start,&one,map+e->gid*l,1);
      }
    }
  
  gather = gs_init(map,l*nes,option("GSLEVEL"));
  gs_gop(gather,edge,"+");
  gs_free(gather);
  free(map);
      
  
  l   = (LGmax-3)*(LGmax-2)/2;
  l   = l*(l+1)/2;
  map = ivector(0,l*nfs);
  
  for(i = 0; i < nel; ++i)
    for(j = 0; j < Nface; ++j){
      f = U[i].face + j;
      if(f->gid < nfs){
	start = Mesh[pllinfo.eloop[f->eid]].face[j].gid*l;
	iramp(l,&start,&one,map+f->gid*l,1);
      }
    }
  
  gather = gs_init(map,l*nfs,option("GSLEVEL"));
  gs_gop(gather,face,"+");
  gs_free(gather);
  free(map);
}

void unreduce (double *x, int n)
{
  int nprocs = numnodes(),
      pid    = mynode(),
      k, i;

  ROOTONLY
    for (k = 1; k < nprocs; k++)
      csend (MSGTAG + k, x, n*sizeof(double), k, 0);
  else
    crecv (MSGTAG + pid, x, n*sizeof(double));
  
  return;
}

void reduce (double *x, int n, double *work)
{
  int nprocs = numnodes(),
      pid    = mynode(),
      k, i;

  ROOTONLY {
    for (i = 0; i < n; i++) work[i] = x[i];
    for (k = 1; k < nprocs; k++) {
      crecv (MSGTAG + k, x, n*sizeof(double));
      for (i = 0; i < n; i++) work[i] += x[i];
    }
    for (i = 0; i < n; i++) x[i] = work[i];    
  } else
    csend (MSGTAG + pid, x, n*sizeof(double), 0, 0);
  
  return;
}

void ifexists(double *in, double *inout, int *n, MPI_Datatype *size){
  int i;
  
  for(i = 0; i < *n; ++i)
    inout[i] = (in[i] != 0.0)? in[i] : inout[i];
  
}

void parallel_gather(double *w, Bsystem *B){
  gs_gop(B->pll->solve,w,"+");
}

void exit_comm(void){
  exit_mpi();
}
#endif

