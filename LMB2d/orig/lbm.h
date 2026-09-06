#ifndef _I_lbm_H_
#define _I_lbm_H_

void savevdf(int v);

float tortuosity(void);
void lbm(void);
void initlbm(std::string);
float volumeflux2d(void);
float getporosity(void);
float geteffectiveporosity(float perc);
float specificsurfacearea(void);
float pinumber(void);
float pinumber2(void);
void exportvtk(void);
void exportvelocity(std::string);

#define RESCALETAU1 1
#define LX (256*RESCALETAU1)
#define LY (256*RESCALETAU1)
#define L0 (4*RESCALETAU1)       	// wymiar charakterystyczny, bloczek
const int W = LX/L0;   	// bezwymiarowa szerokość
const int H = LY/L0;

#define SCALE 1

extern float UCOPY[][LY], VCOPY[][LY];
extern int F[][LY];
extern float U[][LY];
extern float R[][LY];
extern float V[][LY];
extern float df[2][LX][LY][9];
extern const float w[9];
//extern float omega;
extern float fx;
extern int numobstacles;
extern float POROSITY;
extern float mu;
#endif

