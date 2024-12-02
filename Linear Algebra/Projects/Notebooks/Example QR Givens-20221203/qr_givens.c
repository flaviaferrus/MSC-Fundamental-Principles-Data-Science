#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void qrgivens(int m, int n, double **a, double **q);
void givens(double a, double b, double *c, double *s);
void prodd(int m, double **q, double c, double s, int i, int j);
void prode(int n,double **a, double c, double s, int i, int j);
void escriure_matriu(int m, int n, double **a);

void qrgivens(int m, int n, double **a, double **q)
{
/*Descomposicio A=QR, Q ortogonal, R triangular superior
  Obs: la R es retorna en A
  Obs: estrategia de plans coordenats i--j amb i=j-1; i=n-1,...,0. */

int i,j;
double c,s;


for(i=0;i<m;i++){ for(j=0;j<m;j++){ q[i][j]=0;} q[i][i]=1;}
for(j=0;j<n;j++){ for(i=m-1;i>j;i--){
                                        givens(a[i-1][j],a[i][j],&c,&s); /*el·limino Aij*/
                                        prode(n,a,c,s,i,j);
                                        prodd(m,q,c,s,i,j);
				    } }
}

void givens(double a, double b, double *c, double *s)
{
double r;
r=a*a+b*b;  if(r>1.e-32){ r=sqrt(r); *c=a/r; *s=-b/r;} else{*c=1; *s=0;}
}

void prodd(int m, double **q, double c, double s, int i, int j)
{
int k;
double aux;
for(k=0;k<m;k++) { aux=q[k][i-1]; q[k][i-1]=c*aux-s*q[k][i]; q[k][i]=s*aux+c*q[k][i]; }
}

void prode(int n,double **a, double c, double s, int i, int j)
{
int k;
double aux;
for(k=j;k<n;k++){aux=a[i-1][k]; a[i-1][k]=c*aux-s*a[i][k]; a[i][k]=s*aux+c*a[i][k];}
//a[i][j]=0; /*poso a 0 la part inferior de la matriu*/
}

