#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void escriure_matriu(int m, int n, double **a);
void prodmat(int m1,int n1,double **a,int n2, int m2,double **b,double **c);

void qrgivens(int m, int n, double **a, double **q);
void givens(double a, double b, double *c, double *s);
void prodd(int m, double **q, double c, double s, int i, int j);
void prode(int n,double **a, double c, double s, int i, int j);

int main()
{
int i,j;

int m,n; /*n files, m col*/
double **a,**q; /* vull A=QR */
double **AA,**AA1,**Qt,**a_cp,max,aux; 
int igrand;

printf("#genero random? \n"); scanf("%d",&igrand);
printf("#m,n?\n"); scanf("%d %d",&m,&n);

a=(double **) calloc(m,sizeof(double*)); for(i=0;i<m;i++) a[i]=(double *) calloc(n, sizeof(double));
q=(double **) calloc(m,sizeof(double*)); for(i=0;i<m;i++) q[i]=(double *) calloc(m, sizeof(double));  /* Q es m*m*/
a_cp=(double **) calloc(m,sizeof(double*)); for(i=0;i<m;i++) a_cp[i]=(double *) calloc(n, sizeof(double));
AA1=(double **) calloc(m,sizeof(double*)); for(i=0;i<m;i++) AA1[i]=(double *) calloc(n, sizeof(double));
AA=(double **) calloc(m,sizeof(double*)); for(i=0;i<m;i++) AA[i]=(double *) calloc(m, sizeof(double));
Qt=(double **) calloc(m,sizeof(double*)); for(i=0;i<m;i++) Qt[i]=(double *) calloc(m, sizeof(double));

if(igrand==0){for(i=0;i<m;i++){ for(j=0;j<n;j++) {scanf("%lf",&a[i][j]); a_cp[i][j]=a[i][j]; }}}
else{
    srand((unsigned)time(NULL));
    for(i=0;i<m;i++){ for(j=0;j<n;j++){ a[i][j]=((double)rand()/(double) RAND_MAX +1); a_cp[i][j]=a[i][j]; }}
    }

qrgivens(m,n,a,q);

/*check: R triang inferior*/
max=0; for(i=0;i<m;i++) { for(j=0; j<i;j++){ aux= fabs(a[i][j]); if(aux>max) max=aux;}}
printf("# error triang sup %24.16e\n",max);

/*check: A=Q*R*/
prodmat(m,m,q,m,n,a,AA1);
max=0; for(i=0;i<m;i++){ for(j=0;j<n;j++){ aux=fabs(AA1[i][j]-a_cp[i][j]); if(aux>max) max=aux;}}
printf("# error QR-A       %24.16e\n",max);

/*check: Qt*Q=Id*/
for(i=0;i<m;i++){for(j=0;j<m;j++){ Qt[j][i]=q[i][j];}}
prodmat(m,m,Qt,m,m,q,AA);
for(i=0;i<m;i++){ for(j=0;j<n;j++) {aux=fabs(AA[i][j]);if(i==j) aux=aux-1;  if(aux>max) max=aux;}}
printf("# error QtQ-Id     %24.16e\n",max);


return 0;
}

