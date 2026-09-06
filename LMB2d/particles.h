void drawpar(int mode);
void initpar();
void initparprobability(void);
void initparprobabilityINVERSE(void);
void movepar(double dt);
void moveparinterpolate(double dt);

const int NPARMAX = 256*4096;
extern float mnoznik_alpha;
extern float ALPHAPROB;

extern int npar;			// num of particles
extern int PSIZ;
void initrendervbo(void);
extern double probrand_part;
