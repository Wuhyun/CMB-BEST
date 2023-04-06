/*******************************************************************************
 * This file is part of QUADPTS.
 * Copyright (c) 2012 Nick Hale and Alex Townsend
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 ******************************************************************************/

#include <math.h>
#include <stdio.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_math.h>
#include "gl_integration.h"

#define PI 3.141592653589793
#define PI_HI  3.141592741012573242187
#define PI_LO -0.00000008742278000372485127293903
#define EPSILON 2.2204460492503131e-16
#define SQRTEPS 1.4901161193847661e-08
#define M 30
#define ITER_MAX 10

int asy1(double *nodes, double*weights, unsigned long int n);
int feval(int n, double theta, double C, double* f, double* fp, int k, int flag);
int feval_asy1(int n, double theta, double C, double* f, double* fp, int k);
int feval_asy2(int n, double t, double* f, double* fp, int flag);
int mycosA(int n, double theta, double* cosA, double* sinA, int k);
double clenshaw( double* c, int n, double x);
double besseltaylor( double x, double y );
double tB1( double x );
double A2( double x );
double tB2( double x );
double A3( double x );
//int main( void );

int asy(double *nodes, double*weights, unsigned long int n)
{
    int s = n % 2;                       /* True if n is odd */
    unsigned long int n3, n4;
    int k, iter;
    double x, w, t, dt, C, theta, sint, f, fp, S, dS, dn = (double)n;
    double fn, fn5, nk, n5k, pn, jk, num, den, p , pp, err;
    
    /* The constant out the front of the sum in evaluation (compute only once) */
    /* C = sqrt(pi/4)*gamma(n+1)/gamma(n+3/2) */
    /* Use Stirling's series */
    dS = -.125/dn;
    S = dS; 
    k = 1; 
    while ( fabs(dS/S) > EPSILON/100 ) {
        k += 1;
        dS *= -.5 * (k-1) / (k+1) / dn;
        S += dS;
    }
    double stirling[10] = {1., 1./12., 1./288., -139./51840., -571./2488320., 
                           163879./209018880., 5246819./75246796800., 
                          -534703531./902961561600., -4483131259./86684309913600., 
                           432261921612371./514904800886784000.};
    fn = 1; fn5 = 1; nk = dn; n5k = dn+.5;
    for ( k = 1 ; k < 10 ; k++ ) {
        fn += stirling[k]/nk;
        fn5 += stirling[k]/n5k;
        nk *= dn;
        n5k *= dn + .5;
    }
    C = exp(S)*sqrt(4.0/(dn+.5)/PI) * fn / fn5;  
    
    /* Approximations of roots near boundaries */
    double besselroots[32] = {2.404825557695773,  5.520078110286311,
                              8.653727912911013, 11.791534439014280,
                             14.930917708487790, 18.071063967910920,
                             21.211636629879269, 24.352471530749284,
                             27.493479132040250, 30.634606468431990,
                             33.775820213573560, 36.917098353664045,
                             40.058425764628240, 43.199791713176737,
                             46.341188371661815, 49.482609897397822,
                             52.624051841114984, 55.765510755019974,
                             58.906983926080954, 62.048469190227159,
                             65.189964800206866, 68.331469329856787,
                             71.472981603593752, 74.614500643701817,
                             77.756025630388066, 80.897555871137627,
                             84.039090776938195, 87.180629843641128,
                             90.322172637210500, 93.463718781944763,
                             96.605267950996279, 99.746819858680624};
    
    /* Useful constants */                         
    pn = dn*dn+dn+1./3.; 
    n3 = n*n*n; 
    n4 = n3*n; 
       
    /* Loop over each root */
    for ( k = (n+s)/2; k > 0; k-- ) {
        
        /* Initialise */
        dt = 1.0; 
        iter = 1;
        
        /* Asymptotic approximation of roots (Tricomi, 1950) */
        theta = PI*(4.0*k-1)/(4.0*n+2.0);
        sint = sin(theta);
        x = (1.0 - (n-1.0)/(8.0*n3)-(39.0-28.0/(sint*sint))/(384.0*n4))*cos(theta);  
        t = acos(x);
                
        /* Use different guesses near the ends */
        if (x > 0.5) { 
            
            /* Compute some bessel functions */
            if (k < 30) {                   /* These Bessel roots are hard coded */
                jk = besselroots[k-1];          
            }
            else {                          /* Compute more (Branders, JCP 1981) */
                p = (k-0.25)*PI;
                pp = p*p;
                num = 0.0682894897349453 + pp*(0.131420807470708 + pp*(0.0245988241803681 + pp*0.000813005721543268));
                den = p*(1.0 + pp*(1.16837242570470 + pp*(0.200991122197811 + pp*(0.00650404577261471))));
                jk = p + num/den;
            }
            
            /* Evaluate asymptotic approximations */
            if ( k <= 5 ) {                 /* Extreme boundary (Gatteschi, 1967) */
                t = jk/sqrt(pn)*(1.-(jk*jk-1.)/(pn*pn)/360.); 
                dt = 0.0;
            }
            else {                          /* Boundary (Olver, 1974) */
                p = jk/(n+.5);
                t = p + (p/tan(p)-1.)/(8.*p*(n+.5)*(n+.5));
            }
        }
        
        /* Newton iteration for roots */
        err = EPSILON + 1.;
        while (err > SQRTEPS/100) {
            feval(n, t, C, &f, &fp, k,0);     /* Evaluate at current approximation */
            dt = f/fp;                      /* Newton update */
            t += dt;                        /* Update t */
            if (iter++ > ITER_MAX) { break; } 
            err = fabs(dt);
        }
        feval(n, t, C, &f, &fp, k,1);            /* Once more for luck */

        /* Convert back to x-space */
        x = cos(t);
        w = 2./(fp*fp);        
        
        /* Store nodes and weights */
        nodes[n-k] = x;
        nodes[k-1] = -x;
        weights[n-k] = w;
        weights[k-1] = w;
    }
    
    /* Enforce x = 0 for odd n */ 
    if (s == 1) {
        nodes[(n+s)/2-1] = 0;
    }

    return 0;
}

int feval(int n, double theta, double C, double* f, double* fp, int k, int flag)
{
    if (k > 10) {
        feval_asy1(n, theta, C, f, fp, k);
    }
    else {
        feval_asy2(n, theta, f, fp, flag);
    }
    return 0;
}

double tB1( double x )
{
    /* Chebyshev expansion on [0 .5] */
   double coeffs[10] = {0.001894863314419,   0.001931311486971,
   0.000044175868501,   0.000007937411347,   0.000000235861687,
   0.000000026985313,   0.000000000912562,   0.000000000076891,
   0.000000000004309,   0.000000000000955};
   return clenshaw(coeffs, 10, 4.*x-1.);
}

double A2( double x )
{
    /* Chebyshev expansion on [0 .5] */
   double coeffs[10] = {0.000193442222401,   0.000261532806021,
   0.000070849964269,   0.000003201466677,   0.000000460249018,
   0.000000019730880,   0.000000001883998,   0.000000000217267,
  -0.000000000055637,   0.000000000043205};
   return clenshaw(coeffs, 10, 4.*x-1.);
}

double tB2( double x )
{
    /* Chebyshev expansion on [0 .5] */
   double coeffs[10] = {-0.001043953145466,  -0.001086540642303,
   -0.000051890885109,  -0.000009762308109,  -0.000000493161531,
   -0.000000048529058,  -0.000000010061677,   0.000000002475646,
  -0.000000001750018,   0.000000000000022};
   return clenshaw(coeffs, 10, 4.*x-1.);
}

double A3( double x )
{
    /* Chebyshev expansion on [0 .5] */
   double coeffs[10] = {-0.219685228198227,  -0.299245558904199, 
           -0.085543519934549,  -0.006535493269862,  -0.000863987090576, 
           -0.000217345262104,   0.000045325326574,  -0.000049009331669,
           0.000000002078413,  -0.000000001050532};
   return clenshaw(coeffs, 10, 4.*x-1.)/1000;
}

double clenshaw( double* c, int n, double x) {
    /* Clenshaw's rule for Chebyshev evlaution */
    double bk = 0., bk1 = 0., bk2 = 0.;
    int k;

    for (k = n-1; k > 0; k-- ) { 
        bk = c[k] + 2.*x*bk1 - bk2;
        bk2 = bk1; 
        bk1 = bk;
    }
    return c[0] + x*bk1 - bk2;
}


int feval_asy2(int n, double t, double* f, double* fp, int flag) {   
    double dn = (double)n, rho = dn + .5, rho2 = dn - .5;
    double sint, csct, cost, cott, tinv;
    double tB1t = 0., A2t = 0., tB2t = 0., A3t = 0.;
    double Ja, Jab, Jb, Jbb, gt, gtdx, vals, vals2, A1, tB0, denom;
    
    /* Evaluate bessel functions  */
    Ja = gsl_sf_bessel_J0 ( rho*t );
    Jb = gsl_sf_bessel_J1 ( rho*t );
    Jbb = gsl_sf_bessel_J1 ( rho2*t );
    
    if (flag == 1) {
        Jab = besseltaylor(-t,rho*t);   
    }
    else {
        Jab = gsl_sf_bessel_J0 ( rho2*t );
    }

    /* Evaluate functions for recurrsive definition of coefficients. */
    sint = sin(t);
    csct = 1/sint;
    cost = cos(t);
    cott = csct*cost;
    tinv = 1/t;
    gt = 0.5*(cott-tinv);
    gtdx = 0.5*(-csct*csct + tinv*tinv);
    tB0 = 0.25*gt;
    A1 = 0.125*(gtdx-gt*tinv-0.25*gt*gt);   
    
    /* first term: */
    vals = Ja;
    vals2 = Jab;
    
    /* second term: */
    vals = vals + Jb*tB0/rho;
    vals2 = vals2 + Jbb*tB0/rho2;

    /* third term: */
    vals = vals + Ja*A1/(rho*rho); 
    vals2 = vals2 + Jab*A1/(rho2*rho2);
    
    /* higher terms: */
    tB1t = tB1(t);
    vals = vals + Jb*tB1t/(rho*rho*rho); 
    vals2 = vals2 + Jbb*tB1t/(rho2*rho2*rho2); 
    
    A2t = A2(t);
    vals = vals + Ja*A2t/(rho*rho*rho*rho); 
    vals2 = vals2 + Jab*A2t/(rho2*rho2*rho2*rho2);
    
    tB2t = tB2(t);
    vals = vals + Jb*tB2t/(rho*rho*rho*rho2*rho2); 
    vals2 = vals2 + Jbb*tB2t/(rho2*rho2*rho2*rho2*rho2);  
    
    A3t = A3(t);
    vals = vals + Ja*A3t/(rho*rho*rho*rho*rho*rho); 
    vals2 = vals2 + Jab*A3t/(rho2*rho2*rho2*rho2*rho2*rho2);
    
    /* combine */
    denom = sqrt(t*csct);
    *fp = n*(-cost*vals + vals2)*csct*denom;
    *f = vals*denom;
            
    return 0;
}
    
int feval_asy1(int n, double theta, double C, double* f, double* fp, int k)
{    
    double sinT = sin(theta), cosT = cos(theta), cotT = cosT/sinT;
    double cosA, sinA, denom, df, dfp, tmp;
    int m;
    
    /* m = 0 */
    denom = sqrtl(2.0*sinT);
    
    /*alpha = (n + 0.5)*theta - 0.25*PI;
    cosA = cos(alpha);
    sinA = sin(alpha);*/
    mycosA(n, theta, &cosA, &sinA, k);
    
    *f = C * cosA/denom;
    *fp = C * ( 0.5*(cosA*cotT+sinA) + n*sinA) / denom;
    
    /* Loop over m until term is sufficiently small */
    for ( m = 1; m <= M; m++ ) {
        C *= (1.0-0.5/m)*(m-0.5)/(n+m+0.5);
        denom *= 2.0*sinT;
        
        /*alpha += theta - 0.5*PI;
        cosA = cos(alpha);
        sinA = sin(alpha);*/
        
        tmp = cosA*sinT + sinA*cosT;
        sinA = sinA*sinT - cosA*cosT;
        cosA = tmp;
        
        df = C * cosA/denom;
        dfp = C * ( (m+0.5)*(cosA*cotT+sinA) + n*sinA ) / denom;
        
        *f += df;
        *fp += dfp;
        
        if (fabs(df) + fabs(dfp) < EPSILON/100){ break; }
    }
    
    return 0;
    
}

int mycosA(int n, double theta, double* cosA, double* sinA, int k) {
        int j;
        double dh, tmp = 0.0, DH, dh2, lo, hi = theta, fixsgn = (double)(1-2*(k%2));
        double k025 = (k-.25), rho = n+.5, sgn = 1.0, fact = 1.0;         
        
        /* bit shift to get a hi-lo version of theta */
        hi = (double)((float)hi);
        lo = theta - hi;  
        dh = (hi*rho-k025*PI_HI) + lo*rho - k025*PI_LO;
        /* easy way: dh = (n+0.5)*theta-(k-.25)*PI; */
        
        DH = dh; dh2 = dh*dh;
        for ( j = 0; j <= 5; j++ ) {
            tmp += sgn*DH/fact;
            sgn = -sgn;
            fact = fact*(double)((2*j+3)*(2*j+2));
            DH *= dh2;
        }
        *cosA = tmp*fixsgn;
        
        tmp = 0.0; sgn = 1.0; fact = 1.0; DH = 1.0;
        for ( j = 0; j <= 5; j++ ) {
            tmp += sgn*DH/fact;
            sgn = -sgn;
            fact = fact*(double)((2*j+2)*(2*j+1));
            DH *= dh2;
        }
        *sinA = -tmp*fixsgn;
        
    return 0;
}

double besseltaylor(double t, double z)
{
    /* Accurate evaluation of Bessel function for asy2 */
    
    double J[60], S, s, sgn, C = 1.;
    int kmax, j, k, kchoosej;

    /* How many terms to take? */
    kmax = (int) ceil(fabs(log(EPSILON)/log(fabs(t))));
    if (kmax > 30) {
        kmax = 30;
    }

    /* Compute the bessel functions at rho*theta */
    J[kmax] = gsl_sf_bessel_J0(z);
    J[kmax+1] = gsl_sf_bessel_J1(z);
    J[kmax-1] = -J[kmax+1] ;
    sgn = 1.;
    for (k = 2; k < kmax ; k++) {
        J[kmax+k] = gsl_sf_bessel_Jn(k, z);
        J[kmax-k] = sgn*J[kmax+k];
        sgn = -sgn;
    }

    /* Evaluate Taylor series */
    S = J[kmax];
    for (k = 1 ; k < kmax ; k++) {
        s = 0.;
        sgn = -1.;
        kchoosej = 1.;
        C = C*t/2./k;
        for (j = 0 ; j<=k ; j++) {
            sgn = -sgn;
            s += sgn*(double)kchoosej*J[kmax-k+2*j];
            kchoosej *= (k-j);
            kchoosej /= (j+1);
        }
        S += C*s;
    }
    return(S);
    
    }

/* The gateway function */
/*
int main( void )
{
  double *result_roots,*result_weights;
  int n = 30, k;
  
  result_roots = (double *)malloc( n*sizeof(double) );
  result_weights = (double *)malloc( n*sizeof(double) );
  
//  Call the C++ subroutine
  asy(result_roots,result_weights,n);
  
//  Print to screen
  printf("x = \n");
  for ( k=0 ; k<n ; k++ ) {
      printf(" %16.16f\n", result_roots[k]);
  }
  
  printf("w = \n");
  for ( k=0 ; k<n ; k++ ) {
      printf(" %16.16f\n", result_weights[k]);
  }
  
// Free memory
  free(result_roots);
  free(result_weights);
  result_roots = NULL;
  result_weights = NULL;
  
  return(0);
}
*/
