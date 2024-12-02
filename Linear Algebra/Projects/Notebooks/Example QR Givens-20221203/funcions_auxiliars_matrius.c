#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void escriure_matriu(int m, int n, double **a);
void prodmat(int m1,int n1,double **a,int m2, int n2,double **b,double **c);

void escriure_matriu(int m, int n, double **a)
{
int i,j;
for(i=0;i<m;i++)
        {
        for(j=0;j<n;j++) printf("%d %d %24.16e\n",i,j,a[i][j]);
        }

}

void prodmat(int m1,int n1,double **a,int m2, int n2,double **b,double **c)
{
int i,j,k;

if(n1!=m2){ printf("dimension problem\n"); exit(1);}
for(i=0;i<m1;i++){
        for(j=0;j<n2;j++){
                          c[i][j]=0;
                          for(k=0;k<m2;k++) c[i][j]=c[i][j]+a[i][k]*b[k][j];
                          }
                  }
}

