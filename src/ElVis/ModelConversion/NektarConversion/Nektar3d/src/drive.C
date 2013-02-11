/*---------------------------------------------------------------------------*
 *                        RCS Information                                    *
 *                                                                           *
 * $Source: 
 * $Revision:
 * $Date: 
 * $Author:
 * $State: 
 *---------------------------------------------------------------------------*/
// tcew done
#include "nektar.h"
#include <stdio.h>
#include <time.h>

static int    Je;            /* Externals */
static double dt, Re;

void MakeF   (Domain *omega, ACTION act);
static void StartUp (Domain *);

static void P_solve(Element_List *U, Element_List *Uf,Bndry *Ubc,Bsystem *Ubsys,
             		SolveType Stype,int step);
static void V_solve(Element_List *U, Element_List *Uf,Bndry *Ubc,Bsystem *Ubsys,
             		SolveType Stype,int step, Domain *Omega);

#if DEBUG
#define Timing(s) \
 ROOTONLY  { fprintf(stdout,"%s Took %g seconds\n",s,(clock()-st)/cps); \
							       st = clock();}
#else
#define Timing(s) \
 /* Nothing */
#endif

int main(int argc, char *argv[])
{
	Domain      *Omega;             /* Solution Domain            */
	int          step, nsteps;      /* Discrete time (step)       */
	double       time = 0.0;        /* Continuous time            */
	ACTION       WaveProp;          /* Convective form            */
	SLVTYPE      SolveType;         /* Solver type                */
	double       begin_clock,t,cfl;
	double       st, cps = (double)CLOCKS_PER_SEC;
	int          Torder, eid_max;

	init_comm(&argc, &argv);

#ifdef DEBUG
	/*  mallopt(M_DEBUG,1); */
	init_debug();
#endif

	Omega     = PreProcess(argc,argv);

	time      = dparam("STARTIME");
	nsteps    = iparam("NSTEPS");
	WaveProp  = (ACTION) iparam("EQTYPE");
	SolveType = (SLVTYPE) iparam("SLVTYPE");
	Je        = iparam("INTYPE");
	dt        = dparam("DELT");
	step      = 0;
	Re        = 1./dparam("KINVIS");

	dparam_set("t",time);

#ifndef WANNIER

	Omega->U->Terror(Omega->soln[0]);
	Omega->V->Terror(Omega->soln[1]);
	Omega->W->Terror(Omega->soln[2]);
	Omega->P->Terror(Omega->soln[3]);
#endif

	StartUp (Omega);
	Omega->dt = dt;

	if (option ("SurForce") == 1)
		printf("Yes, I am dumping sectional forces\n");

    if (option ("SurInflow") == 1) 
	   printf("Yes, I am dumping inflow surface\n");
	
	if(!step)
	{ // impose boundary conditions
		SetBCs(Omega->U,Omega->Ubc,Omega->Usys);
		SetBCs(Omega->V,Omega->Vbc,Omega->Vsys);
		SetBCs(Omega->W,Omega->Wbc,Omega->Wsys);
	}

	forces(Omega,step,time);
	
	  if (option ("SurInflow") == 1) 
	{	
	surf_inflow(Omega,step,time); 
}
	/*------------------------------------------------------------------*
	 *                    Main Integration Loop                         *
	 *------------------------------------------------------------------*/
	begin_clock = (double) clock();

	switch(SolveType)
	{
		case Splitting:
			while (step < nsteps)
			{
				set_order(((step+1 < Je)? step+1: Je));

				st = clock();
				MakeF (Omega, Prep);
				cfl = cfl_checker(Omega,dt);
				ROOTONLY fprintf(stdout,"CFL: %lf\n",cfl);
				cfl = full_cfl_checker(Omega,dt,&eid_max);
				ROOTONLY fprintf(stdout,"Full CFL: %lf in eid %d\n",cfl,
								 eid_max+1);
				Timing("Prep.......");

				MakeF (Omega, WaveProp);
				Timing("WavePropo..");

				// set v^{n+1} before pressure BC's determined
				if(Omega->Tfun && Omega->Tfun->type == Womersley)
                                    SetWomSol(Omega,time+dt,1);

				if(option("tvarying"))
				{
					Bndry *Bc;
					t = dparam("t");
					dparam_set("t",t+dt);
					for(Bc=Omega->Ubc;Bc;Bc=Bc->next)
                                            if(Bc->usrtype != 'w') //not Womersley case
						Bc->elmt->update_bndry(Bc,1);
					for(Bc=Omega->Vbc;Bc;Bc=Bc->next)
                                            if(Bc->usrtype != 'w') //not Womersley case
						Bc->elmt->update_bndry(Bc,1);
					for(Bc=Omega->Wbc;Bc;Bc=Bc->next)
                                            if(Bc->usrtype != 'w') //not Womersley case
						Bc->elmt->update_bndry(Bc,1);
				}

				SetPBCs(Omega);

				Integrate_SS(Je, dt, Omega->U, Omega->Uf, Omega->u, Omega->uf);
				Integrate_SS(Je, dt, Omega->V, Omega->Vf, Omega->v, Omega->vf);
				Integrate_SS(Je, dt, Omega->W, Omega->Wf, Omega->w, Omega->wf);
				Timing("Integrate..");

				MakeF (Omega, Pressure);
				Timing("Pressure...");

				P_solve (Omega->P,Omega->Pf,Omega->Pbc,Omega->Pressure_sys,
				         Helm,step);
				Timing("P_solve....");

				MakeF (Omega, Viscous);
				Timing("Viscous....");

				V_solve(Omega->U,Omega->Uf,Omega->Ubc,Omega->Usys,Helm,step,
						Omega);
				V_solve(Omega->V,Omega->Vf,Omega->Vbc,Omega->Vsys,Helm,step,
						Omega);
				V_solve(Omega->W,Omega->Wf,Omega->Wbc,Omega->Wsys,Helm,step,
						Omega);

				Timing("V_solve....");

				MakeF (Omega, Post);
				Timing("Post.......");

				Analyser(Omega, ++step, (time += dt));
				Timing("Analyser...");
			}
			break;
		case SubStep:
			{
				while (step < nsteps)
				{
					Omega->step = step;

					st = clock();

					MakeF (Omega, Prep);
					Timing("Prep.......");

					Torder = (step+1 < Je)? step+1: Je;
					set_order(Torder);


					SubCycle(Omega,Torder);
					Timing("SubCycle...");

					MakeF (Omega, Pressure);
					Timing("Pressure...");

					P_solve (Omega->P,Omega->Pf,Omega->Pbc,Omega->Pressure_sys,
					         Helm,step);
					Timing("P_solve....");

					MakeF (Omega, Viscous);
					Timing("Viscous....");

					V_solve(Omega->U,Omega->Uf,Omega->Ubc,Omega->Usys,Helm,step,
							Omega);
					V_solve(Omega->V,Omega->Vf,Omega->Vbc,Omega->Vsys,Helm,step,
							Omega);
					V_solve(Omega->W,Omega->Wf,Omega->Wbc,Omega->Wsys,Helm,step,
							Omega);

					Timing("V_solve....");

					MakeF (Omega, Post);
					Timing("Post.......");

					Analyser(Omega, ++step, (time += dt));
					Timing("Analyser...");
				}
			}
			break;
	}
	printf("User time of solver (seconds):  %lf \n",
	       (clock()-begin_clock)/(double)CLOCKS_PER_SEC);

#ifdef WOMERR

	if(Omega->Tfun && Omega->Tfun->type == Womersley)
		WomError(Omega,time);
#else

	Omega->U->Terror(Omega->soln[0]);
	Omega->V->Terror(Omega->soln[1]);
	Omega->W->Terror(Omega->soln[2]);
	Omega->P->Terror(Omega->soln[3]);
#endif

	PostProcess(Omega, step, time);

	exit_comm();

	return 0;
}

static void P_solve(Element_List *U, Element_List *Uf,
             Bndry *Ubc, Bsystem *Ubsys, SolveType Stype, int step)
{
	int Nrhs;
	
	if (step+1 < Je)
	{
		Nrhs = option("MRHS_NRHS");
		option_set("MRHS_NRHS",0);
	}
	
	SetBCs (U,Ubc,Ubsys);
	Solve  (U,Uf,Ubc,Ubsys,Stype);

	if (step+1 < Je)
		option_set("MRHS_NRHS",Nrhs);
}

static void V_solve(Element_List *U, Element_List *Uf, Bndry *Ubc,
					Bsystem *Ubsys, SolveType Stype, int step, Domain *Omega)
{
	int Nrhs;
	
	if (step+1 < Je)
	{
		Nrhs = option("MRHS_NRHS");
		option_set("MRHS_NRHS",0);
	}
	
	if(step && step < Je)
	{
		int k;
		for(k=0;k<U->nel;++k)
			Ubsys->lambda[k].d = Re*getgamma()/dt;

		switch(U->fhead->type)
		{
			case 'u':
				Bsystem_mem_free(Ubsys);
				goto ReCalcMat;
			case 'v':
				if(option("REFLECT1")||option("REFLECT0"))
				{
					if(Ubsys->smeth == direct)
						Bsystem_mem_free(Ubsys);
					else
						Ubsys->Gmat = Omega->Usys->Gmat;
					goto ReCalcMat;
				}
				else
					break;
			case 'w':
				if(option("REFLECT2")||
									(option("REFLECT0")&&option("REFLECT1")&&
									 (!option("REFLECT2"))))
				{
					if(Ubsys->smeth == direct)
						Bsystem_mem_free(Ubsys);
					else
						Ubsys->Gmat = Omega->Usys->Gmat;
					goto ReCalcMat;
				}
				else
					break;
ReCalcMat:			 
				if(Ubsys->Precon == Pre_LEnergy)	// don't recalc precon
					option_set("ReCalcPrecon",0); 	// at present
				double *save_hj = dvector(0, U->hjtot-1);
				fprintf(stdout,"Regenerating %c-Matrix\n",U->fhead->type);
				dcopy(U->hjtot, U->base_hj, 1, save_hj, 1);
				GenMat(U,Ubc,Ubsys,Ubsys->lambda,Helm);
				dcopy(U->hjtot, save_hj, 1, U->base_hj, 1);
				free(save_hj);
				break;
		}

		if(Ubc&&Ubc->DirRHS)
		{
			free(Ubc->DirRHS);
			Ubc->DirRHS = (double*) 0;
		}

#ifndef PARALLEL
		if(!option("tvarying"))
			DirBCs(U,Ubc,Ubsys,Helm);
#endif

	}

	SetBCs (U,Ubc,Ubsys);
	Solve  (U,Uf,Ubc,Ubsys,Stype);
	
	if (step+1 < Je)
		option_set("MRHS_NRHS",Nrhs);
}

void MakeF(Domain *omega, ACTION act)
{

	Element_List  *U    =  omega->U,  *V    =  omega->V,   *W   = omega->W,
	              *Uf   =  omega->Uf, *Vf   =  omega->Vf,  *Wf  = omega->Wf,
	              *P    =  omega->P,  *Pf   =  omega->Pf;

	int Nmodes = U->hjtot, Nquad = U->htot;

	switch (act)
	{
		case Prep: /* put fields in physical space for waveprop */
			U->Trans(U,J_to_Q);
			V->Trans(V,J_to_Q);
			W->Trans(W,J_to_Q);
			U->Set_state('p');
			V->Set_state('p');
			W->Set_state('p');

			dcopy(U->htot,U->base_h,1,omega->u[0],1);
			dcopy(V->htot,V->base_h,1,omega->v[0],1);
			dcopy(W->htot,W->base_h,1,omega->w[0],1);
			break;

		case Rotational:
			VxOmega (omega);
			goto AddForcing;

		case Convective:
			VdgradV (omega);
			goto AddForcing;

		case Stokes:
			//    StokesBC (omega);
			Uf->zerofield();
			Vf->zerofield();
			Wf->zerofield();
			Uf->Set_state('p');
			Vf->Set_state('p');
			Wf->Set_state('p');
			goto AddForcing;

AddForcing:
			{
				double fx = dparam("FFX");
				double fy = dparam("FFY");
				double fz = dparam("FFZ");

				if(fx)
					dsadd(Nquad, fx, Uf->base_h, 1, Uf->base_h, 1);
				if(fy)
					dsadd(Nquad, fy, Vf->base_h, 1, Vf->base_h, 1);
				if(fz)
					dsadd(Nquad, fz, Wf->base_h, 1, Wf->base_h, 1);

				if(omega->ForceFuncs)
				{
					dvadd(Nquad, omega->ForceFuncs[0], 1, Uf->base_h, 1,
						  Uf->base_h, 1);
					dvadd(Nquad, omega->ForceFuncs[1], 1, Vf->base_h, 1,
						  Vf->base_h, 1);
					dvadd(Nquad, omega->ForceFuncs[2], 1, Wf->base_h, 1,
						  Wf->base_h, 1);
				}

				break;
			}

		case Pressure:
			{
				double    dtinv = 1./dt;
				double *nul= (double*)0;

				U->Grad_d(Pf->base_h, nul, nul, 'x');

				V->Grad_d(nul, P->base_h, nul, 'y');
				dvadd(Nquad, P->base_h, 1, Pf->base_h, 1, Pf->base_h, 1);

				W->Grad_d(nul,  nul,  P->base_h, 'z');
				dvadd(Nquad, P->base_h, 1, Pf->base_h, 1, Pf->base_h, 1);

				Pf->Set_state('p');

				Pf->Iprod(Pf);

				dscal(Nmodes, dtinv, Pf->base_hj, 1);
				Pf->Set_state('t');

				break;
			}

		case Viscous:
			{
				double dtinv = 1/dt;

				P->Trans(P, J_to_Q);
				P->Set_state('p');
				P->Grad_d (Uf->base_h,Vf->base_h,Wf->base_h,'a');
				P->Set_state('t');

				daxpy(Nquad, -dtinv, U->base_h, 1, Uf->base_h, 1);
				Uf->Set_state('p');
				Uf->Iprod(Uf);
				dscal(Nmodes,    Re, Uf->base_hj, 1);
				Uf->Set_state('t');

				daxpy(Nquad, -dtinv, V->base_h, 1, Vf->base_h, 1);
				Vf->Set_state('p');
				Vf->Iprod(Vf);
				dscal(Nmodes,    Re, Vf->base_hj, 1);
				Vf->Set_state('t');

				daxpy(Nquad, -dtinv, W->base_h, 1, Wf->base_h, 1);
				Wf->Set_state('p');
				Wf->Iprod(Wf);
				dscal(Nmodes,    Re, Wf->base_hj, 1);
				Wf->Set_state('t');

				break;
			}
		case Post:
			{

				break;

			}
		default:
			error_msg(MakeF--unknown type of action);
			break;
	}

	return;
}

/* Do initial time step assuming for case where startup field is in physical *
 * space or copy V field to Vs if it is a restart                           */

static void StartUp(Domain *omega)
{

	if(omega->U->fhead->state == 'p')
	{
		ROOTONLY fprintf(stdout,"Locally transforming initial conditions\n");
		omega->U->Trans(omega->U,Q_to_J);
		omega->V->Trans(omega->V,Q_to_J);
		omega->W->Trans(omega->W,Q_to_J);

		omega->U->Set_state('t');
		omega->V->Set_state('t');
		omega->W->Set_state('t');
		omega->P->Set_state('t');
	}
}

