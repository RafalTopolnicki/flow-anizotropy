#include <iostream>
#include <fstream>
#include <cmath>
#ifdef GL
#include <GL/glut.h>
#endif
#include "mt19937ar.h"
#include <Magick++.h> 

#include "lbm.h"
#ifdef GL
#include "particles.h"
#endif
using namespace std;
using namespace Magick; 

float dt=1;
int mode=1;		// draw mode
int pause=0;
float mnoznik_alpha=0.1;
int interponoff = 1;			// interpolate velocity?
int visualization=1;

int srandused = 1000;

float phi_real=0;//kit

int finished = 0;
const float eps = 0.001;
float delta = 10;
static int step=0;				// 0-1001 bez zerowania
static float flux1=1e10,flux2=-1e10;
const int MINIMUMSTEPS = 1000;


void idleFunction(std::string filename, std::string resultfilename, std::string velocityfilename)
{
	//exit(0);
	static int s = 0;
    float T1,T2;
	const int STEPTOCHECK = 50;		//5000
  const int STEPTOWRITE = 1;

   if(pause==0)
   {		

    //T1 = tortuosity();

    lbm();       // kryterium stop
    s++;
    // kryterium stop
      step++;
      delta = 10;
      if(step==1)
    	    flux1 = volumeflux2d();
    	else
    	{
    		if(step%STEPTOCHECK==0)
    		{
    		    flux2 = volumeflux2d();

    		    if(std::isnan(flux2) || std::isinf(flux2))
    		    {
    		        cerr << "NaN/Inf detected in volume flux at step " << s << " — aborting." << endl;
    		        finished = 1;
    		        exit(2);
    		    }

		        float k = flux2 * mu / fx;

        if(flux2)
          delta = (fabs(flux1-flux2)/fabs(flux2));

		   cout << s << " " << k << " " << tortuosity() << " "<< delta <<endl;

		   flux1 = flux2;
    		}
    	}
  	if(s > MINIMUMSTEPS && delta < eps)
      finished = 1;

   }

   static int g=0;  // limit saving when finished=1

   if(finished==1)
   {
        fstream file2(resultfilename, std::ofstream::out | std::ofstream::app);
            float k = volumeflux2d() * mu / fx;
            file2 << filename << " " << fx << " " << s << " " << k << " " << tortuosity() << " "<< delta <<endl;
        file2.close();
        exportvelocity(velocityfilename);
   }

  // file.close();



}

void init(std::string filename, std::string resultfilename)
{
	cout << "# initialize lbm loop "<< endl;
	cout << fx << endl;
	initlbm(filename); // INIT LBM
    
	//for(int i=0; i<1e2; i++) lbm();
    cout << "# done" << endl;
}


extern int STARTX;
extern int ENDX;


int main(int argc, char**argv)
{
  //cout << "./ POROSITY FORCE SRAND" << endl;
  //exit(0);
  POROSITY = 0.6;//95;
  srandused = 1001;

   // no loop
  if(argc!=5)
  {
    cerr << "Wrong number of input parameters" << endl;
    cerr << "input_filename append_filename velocity_filename force_factor" << endl;
    return -1;
  }
  init_genrand(srandused);
  srand(srandused);
  //domyslne
  fx = 2.5e-07;
  float fx_factor = std::stof(argv[4]);
  fx = fx * fx_factor;
  cout << "Srand: " << srandused << endl;
  cout << "REMEMBER TO CHECK MARGINS OF PI COMPUTATION(!!) ->>" << endl;
  cout << "STARTX = " << STARTX << " " << "ENDX = " << ENDX << endl;
  cout << "Input structure file. PPM or DAT" << argv[1] << endl;
  cout << "Results will be appended to " << argv[2] << endl;
  cout << "Velocity will be saved to " << argv[3] << endl;
  cout << "!! External force factor " << fx_factor << endl;
  cout << "!! External force " << fx << endl;


  init(argv[1], argv[2]); // INIT LBM

  // Initialise ImageMagick library
  InitializeMagick(*argv);

  while(!finished)
  {
      idleFunction(argv[1], argv[2], argv[3]);
      static int ss=0;
  }

return 0;
}
