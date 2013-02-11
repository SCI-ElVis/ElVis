/*
 * proto.h
 *
 * This file conatins function prototypes
 *
 * Started 8/27/94
 * George
 *
 * $Id: proto.h,v 1.3 2008-11-17 17:54:33 bscarmo Exp $
 *
 */


/* entrypoint.o */
int PMETIS(int *, int *, int *, int *, int *, int *, int *, int *, int *,
		   int *, int *);
int KMETIS(int *, int *, int *, int *, int *, int *, int *, int *, int *,
		   int *, int *);
int OMETIS(int *, int *, int *, int *, int *, int *, int *);
