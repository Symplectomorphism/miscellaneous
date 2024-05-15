#ifndef SUPERFLUOUS_H_ 
#define SUPERFLUOUS_H_

void quickSort( double a[], int l, int r, int o[]);
int partition( double a[], int l, int r, int o[]) ;
double cross( double g[], int n, double delta, int w); 
double Random(void);
void PlantSeeds(long x);
void PutSeed(long x);
void GetSeed(long *x);
void SelectStream(int index);
void TestRandom(void);
double Uniform(double a, double b);
void sample( int s[], int n);
void crops_p(double MM[], int nn, int mm, double mu[], double delta, int o_init[], int gg_i[], int gg_j[], int gg_k[], int w);

#endif
