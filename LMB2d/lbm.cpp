
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <Magick++.h> 
using namespace std;
using namespace Magick; 
#include "lbm.h"

// MARGINS
//const float FRACTION = 0.333;
const int dx = LX/2;		////FRACTION*LX/2.0;
int STARTX = LX/2 - dx;
int ENDX =   LX/2 + dx;

// A1
float df[2][LX][LY][9];   

const int ex[9] = {0,  1,0,-1, 0,  1,-1,-1, 1};
const int ey[9] = {0,  0,1, 0,-1,  1, 1,-1,-1};
const int inv[9] = {0, 3,4, 1, 2,  7, 8, 5, 6};
const float w[9]={4.0/9.0,  1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,   1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0};

float UCOPY[LX][LY], VCOPY[LX][LY];
float U[LX][LY], V[LX][LY], R[LX][LY];
int F[LX][LY];
int F_EFF[LX][LY];          // effective field


//float fx = 0.00036864; //
//float fx = 0.000001;

//fx jest ustawiany w funkcji main()
//wartosc domyslna 2.5e-07
float fx = 0.0; // 2.5e-07;//1.6e-05;
float fy = 0.0;   // ustawiany w main(), jak fx

float tau = 1;            // 0.68, vis = (2tau-1)/6 approx 0.167
float mu = 0.33*(tau-0.5);        // vlbm=cs2 ( τ – 0.5 ) dt, 0.0594 dla tau=0.68
                              // 0.33 * 0.05 = 0.0165

//const int RADIUS = 2;
int numobstacles=0;       // from generation procedure
float POROSITY=0.6; // init from user

float pinumber2(void)
{
    float sumej = 0;
    int nodes = 0;
    int volume=0;     
    //for(int i=0; i< LX; i++)
    for(int i=STARTX ; i < ENDX; i++)
    for(int j=0; j< LY; j++)
    {
      if(F[i][j] == 0)
      {
        sumej = sumej + U[ i ][ j ]*U[ i ][ j ] + V[ i ][ j ]*V[ i ][ j ];
        nodes++;
      }          
      volume++;
    }  
    float average_u = sumej / nodes;

    float nsumqisquare = 0;
    float q2;
    //for(int i=0; i< LX; i++)
    for(int i=STARTX ; i < ENDX; i++)
    for(int j=0; j< LY; j++)
    {
      if(F[i][j] == 0)
      {
          float ei = U[ i ][ j ]*U[ i ][ j ] + V[ i ][ j ]*V[ i ][ j ];
          q2 = (ei*ei)/(sumej*sumej);
          nsumqisquare += q2;
      }
    }
    float pi = 1.0f / (nodes * nsumqisquare);
    return pi;
}

float pinumber(void)
{
    float sumej = 0;
    int nodes = 0;  



    for(int i=STARTX; i < ENDX; i++)		// oblicz sume do rownania (4.72)
    for(int j=0; j < LY; j++)
      if(F[i][j] == 0)
      {
        sumej = sumej + U[ i ][ j ]*U[ i ][ j ] + V[ i ][ j ]*V[ i ][ j ];
        nodes++;
      }          

    float average_u = sumej / nodes;
    float nsumqisquare = 0, qi2;
    for(int i=STARTX; i< ENDX; i++)
    for(int j=0; j< LY; j++) 
    if( F[ i ][ j ] == 0 )
    {
          float ei = U[ i ][ j ]*U[ i ][ j ] + V[ i ][ j ]*V[ i ][ j ];
          qi2 = (ei*ei)/(sumej*sumej);
          nsumqisquare += qi2;
    }

    float pi = 1.0f / (nodes * nsumqisquare);			// 4.71
    return pi;
}

float volumeflux2d(void)
{
  // oblicz strumień objętości (z pzekroju wzdłuż osi y), równanie 4.65

  float Q = 0;          // objętościowe natężenie przepływu (4.64)
  float dy = 1.0 / L0;  // H/LY
  float dA = dy;        // długość w 2D
  float A = L0;         // pole powierzchni przekroju (długość w 2D)
  int i=LX/2;           // pozycja przekroju
  for(int j=0; j<LY; j++)
  {
    Q += U[i][j] * dA;
  }

  float q = Q / A;
  return q;
}


// ---------------------------------------------------------------------------
// Superficial (Darcy) flux averaged over the whole periodic cell.
//
// The single-cross-section volumeflux2d() above is kept for backward
// compatibility, but it measures only u_x on the column i=LX/2 and is
// therefore useless both as a convergence monitor and as the transverse
// component of the permeability tensor.  These averages use every node
// (solid nodes hold u=0), and reduce to the same normalisation as
// volumeflux2d(): q_i = sum(u_i) / (L * L0^2).
// ---------------------------------------------------------------------------
void volumeavg(double &qx, double &qy)
{
  double su = 0.0, sv = 0.0;
  for(int i=0; i<LX; i++)
  for(int j=0; j<LY; j++)
  {
    su += U[i][j];
    sv += V[i][j];
  }
  qx = su / (double(LX) * double(L0) * double(L0));
  qy = sv / (double(LY) * double(L0) * double(L0));
}

// Whole-field L2 residual against the last snapshot.  Diagnostic only:
// it is dominated by pointwise magnitudes and is a poor proxy for the
// convergence of the (heavily cancelling) transverse mean, so it must not
// be used as a stopping criterion.
static float USNAP[LX][LY], VSNAP[LX][LY];

void snapshot_field(void)
{
  for(int i=0; i<LX; i++)
  for(int j=0; j<LY; j++)
  {
    USNAP[i][j] = U[i][j];
    VSNAP[i][j] = V[i][j];
  }
}

float field_residual(void)
{
  double num = 0.0, den = 0.0;
  for(int i=0; i<LX; i++)
  for(int j=0; j<LY; j++)
  {
    double du = U[i][j] - USNAP[i][j];
    double dv = V[i][j] - VSNAP[i][j];
    num += du*du + dv*dv;
    den += double(U[i][j])*U[i][j] + double(V[i][j])*V[i][j];
  }
  return den > 0.0 ? float(sqrt(num/den)) : 10.0f;
}

// Porosity over the full periodic cell.  getporosity() above skips the
// j=0 and j=LY-1 rows, which was correct when those rows were no-slip
// walls; the structure reader overwrites them, so the cell is periodic in
// both directions and every node counts.
float getporosity_full(void)
{
  int nfluid = 0;
  for(int i=0; i<LX; i++)
  for(int j=0; j<LY; j++)
    if(F[i][j] == 0) nfluid++;
  return nfluid / float(LX*LY);
}

// Restore the distribution functions to rest without re-reading the
// structure, so the second forcing direction starts from the same state
// the first one did.  F[][] is deliberately left alone.
void resetdistributions(void)
{
  for(int i=0; i<LX; i++)
  for(int j=0; j<LY; j++)
  {
    for(int k=0; k<9; k++)
      df[0][i][j][k] = df[1][i][j][k] = w[k];
    U[i][j] = V[i][j] = 0.0f;
    R[i][j] = 1.0f;
  }
  lbm_reset_parity();
}

float geteffectiveporosity(float perc)
{  
// velocity sum
	float uaverage=0;
	int nf = 0;
	//for(int i=0; i<LX; i++)
  for(int i=STARTX ; i < ENDX; i++)
	for(int j=0; j<LY; j++)
	{
        if(F[i][j] == 0)
        {
            nf++;  
    		uaverage = uaverage + sqrt(U[i][j]*U[i][j]+V[i][j]*V[i][j]);    
        }
    }
	uaverage /= nf;


    // make eff_F
    //for(int i=0; i<LX; i++)
  for(int i=STARTX ; i < ENDX; i++)
    for(int j=0; j<LY; j++)		
    {
        if(F[i][j] == 1)        // flaga
            F_EFF[i][j] = 1;    // sciana
        else
        if( sqrt(U[i][j]*U[i][j]+V[i][j]*V[i][j]) > (perc/100.) * uaverage )         // odrzuc 10% najwolniejszych komorek
            F_EFF[i][j] = 0;
        else
            F_EFF[i][j] = 1;        // wolne predkosci nie wchodza do porowatosci..
    }

    //     
    int nfluid=0;
    //for(int i=0; i<LX; i++)
    for(int i=STARTX ; i < ENDX; i++)
    for(int j=1; j<LY-1; j++)		// uwzgledniamy sciany - gorna i dolna...
    {
        if(F_EFF[i][j] == 0)        // flaga
                 nfluid++;
    }
  return nfluid/float((ENDX-STARTX)*(LY-2));
}


float getporosity(void)
{
    int nfluid=0;
    //for(int i=0; i<LX; i++)
    for(int i=STARTX ; i < ENDX; i++)
    for(int j=1; j<LY-1; j++)		// uwzględniamy ściany - górną i dolną
    if(F[i][j] == 0)
      nfluid++;
  return nfluid/float((ENDX-STARTX)*(LY-2));
}


float specificsurfacearea(void)
{
    //float p=0;
    int walls=0;
    for(int i=STARTX ; i < ENDX; i++) //for(int i=1; i<LX-1; i++)
    for(int j=1; j<LY-1; j++)
      if(F[i][j] == 0)        // komórka z płynem
      {
        if(i-1!=0 && 	F[i-1][j]) walls++;
        if(i+1!=LX-1 && F[i+1][j]) walls++;
        if(j-1!=0 &&	F[i][j-1]) walls++;
        if(j+1!=LY-1 &&	F[i][j+1]) walls++;
      }
  return walls;
}

float tortuosity(void)
{
 /* // Skrypt Palabos
  T calc_tortuositypal(MultiBlockLattice2D<T, DESCRIPTOR> &lat)
   {
       Box2D crossection(0,0,0,L-1);
       plint xComponent = 0;
       //T q = computeSum(*computeVelocityComponent(lat, crossection, xComponent ) );
       Box2D pm(MAR,MAR+L-1,0,L-1);
       T absvelsum = computeSum(*computeVelocityNorm(lat, pm) );
       T xvelsum = computeSum(*computeVelocityComponent(lat, pm, xComponent ));
       //T t = absvelsum / ( q * L ) ;
       T t = absvelsum / xvelsum;
       return t;
   }
   */

   // calc absvelsum
   float absvelsum = 0, xvelsum = 0;
   
   for(int i=STARTX ; i < ENDX; i++)
   for(int j=0 ; j < LY ; j++)
   if(F[ i ][ j ] == 0)
   {
    absvelsum += sqrt(U[ i ][ j ]*U[ i ][ j ] + V[ i ][ j ]*V[ i ][ j ]);
      xvelsum += U[ i ][ j ];
   }        
   
   float t=0;
   if(xvelsum) t = absvelsum / xvelsum;
  
   return t;   
}

// generate random system at given porosity
void generateRAN(float por)
{
    //for(int k = 0; k<10; k++)
    float porosity = 1;//1-LX*2/float(LX*LY);

    // horizontal wall
    for(int i=0;i<LX;i++)
    {
    	F[i][0] = 1;   
    	F[i][LY-1] = 1;   
    }

    while(porosity > por)
    {
    	int x0 = STARTX+(LX-STARTX*2)*(rand()/float(RAND_MAX));
       	int y0 = LY*(rand()/float(RAND_MAX));
       	numobstacles++;
            
        for(int i=x0-L0/2; i < x0+L0/2; i++)
        for(int j=y0-L0/2; j < y0+L0/2; j++)
        {
         	 //if( (i-x0)*(i-x0)+(j-y0)*(j-y0) < RADIUS*RADIUS )
           //if(abs(i-x0) <= L0 && abs(j-y0) <= L0)
         //{
                int ip = (i + LX) % LX;
                int jp = (j + LY) % LY;
    	        
                if(F[ip][jp] == 0)
                {
	    	          F[ip][jp] = 1;
	    	    }
          //}
        }
        porosity = getporosity();
        cout << getporosity() << endl;
    }
    


    // test porosity
    //cout << "Porosity difference: " << getporosity() - porosity << endl;
}

void generate11(float por)					// generate simple random 1x1 system (for percolation threshold)
{
//	fstream file3("results.dat", std::ofstream::out | std::ofstream::app);


  	while(getporosity() > por)
    {
   		int x0 = LX * ( rand()/float(RAND_MAX) );
   		int y0 = LY * ( rand()/float(RAND_MAX) );
   		
        if(F[x0][y0]==0)
        {
        	numobstacles++;
			F[x0][y0] =1;
		}
	
		cout << getporosity() << " " << specificsurfacearea() << " " << numobstacles << endl;
//		file3 << getporosity() << " " << specificsurfacearea() << " " << numobstacles << endl;
    }
	cout << "Porosity: " << getporosity() << endl;
//	file3.close();
}

void generate22(float por)
{
  	while(getporosity() > por)
    {
    		int x0 = L0 * int(floor(LX * ( rand()/float(RAND_MAX) ) / L0 ));
    		int y0 = L0 * int(floor(LY * ( rand()/float(RAND_MAX) ) / L0 ));
    		numobstacles++;
            
			for(int i=x0; i < x0+L0; i++)
			for(int j=y0; j < y0+L0; j++)
			{
				
				
					int ip = (i + LX) % LX;
					int jp = (j + LY) % LY; 	        
					if(F[ip][jp] == 0)
						F[ip][jp] = 1;

 			}
     	cout << "Porosity: " << getporosity() << endl;
     }
	
}

// FROM PPM or DAT or GIF
void read_from_gif(std::string filename)
{
  std::cout << "Reading GIF file: " << filename << std::endl;
  Magick::InitializeMagick(nullptr);
  Image image(filename);
  PixelPacket *pixels = image.getPixels(0, 0, LY, LX);
  for(int y=0; y<LY; y++)
    for(int x=0; x<LX; x++)
    {
      if(pixels[LX*y + x].red > 0)
        F[x][y] = 0;
      else
        F[x][y] = 1;
    }
}
void read_from_ppm(std::string filename)
{
  int i,j;
    std::cout << "Reading PPM file: " << filename << std::endl;
    ifstream ppmfile(filename);
    string line;
    // move the header
    getline (ppmfile,line);     // P3
    getline (ppmfile,line);     // NX, NY
    getline (ppmfile,line);     // # Created by IrfanView 
    getline (ppmfile,line);     // depth
      
    for(j=0; j<LY; j++)
    for(i=0; i<LX; i++)        // read data
    {
      ppmfile >> line;            // R
      //F[i][j] = 1-atoi(line.c_str());
      if(atoi(line.c_str()))
        F[i][j] = 0;
      else
        F[i][j] = 1;
      ppmfile >> line; ppmfile >> line;   // G,B
    }

    // close file
    ppmfile.close();
}

void read_from_dat(std::string filename)
{
  int i,j;
    std::cout << "Reading DAT file: " << filename << std::endl;
    ifstream ppmfile(filename);
    string line;
     
    for(j=0; j<LY; j++)
    for(i=0; i<LX; i++)
    {
      ppmfile >> line;
      if( atoi(line.c_str()) == 0)
        F[i][j] = 0;
      else
        F[i][j] = 1;
    }
    // close file
    ppmfile.close();
}

void generate(std::string filename)
{
  if(filename.substr(filename.size() - 4) == ".ppm")
    read_from_ppm(filename);
  else if (filename.substr(filename.size() - 4) == ".gif")
    read_from_gif(filename);
  else
    read_from_dat(filename);
}








// A2
void initlbm(std::string filename)
{
    // inicjalizacja df
	for(int i=0; i < LX ; i++)	
	for(int j=0; j < LY ; j++)	
	for(int k=0; k< 9; k++)	
		df [0][ i ][ j ][k] = df [1][ i ][ j ][k] = w[k];
	
	// czyœæ flagi (ustaw oznaczenia na siatce na 0 - "wszystkie z p³ynem")
	for(int i=0; i < LX; i++)
	for(int j=0; j < LY; j++)
	{
		F[i][j] = 0;
		if(j==0 || j==LY-1)	F[i][j] = 1;		// top/bottom walls
	}
  
	generate(filename);

}



// A4 - wartosci makroskopowe
void macro(int c)
{
	for(int i=0 ; i < LX; i++)
	for(int j=0 ; j < LY; j++)
	if(F[ i ][ j ] == 0)
	{
       	float rho=0,ux=0,uy=0;
   	   	for(int k=0; k<9; k++)			// calculate density and velocity
		{
			rho = rho 	+ df[c][ i ][ j ][ k ];
			ux =  ux 	+ df[c][ i ][ j ][ k ] * ex[ k ];
			uy =  uy 	+ df[c][ i ][ j ][ k ] * ey[ k ];
		}
		ux /= rho;
		uy /= rho;	 
     
		U[ i ][ j ] = ux;
		V[ i ][ j ] = uy;
		R[ i ][ j ] = rho;
    }
 }


// A5 - oblicz "f z tyldą" - krok kolizji
 void collision(int c)
 {
     float ux, uy, rho;
 	for(int i=0 ; i < LX ; i++)
	for(int j=0 ; j < LY ; j++)
	if(F[ i ][ j ] == 0)
 	{	   
       	ux = U[ i ][ j ];
       	uy = V[ i ][ j ];
       	rho = R[ i ][ j ];
		
	    // siła zewnętrzna
 		ux = ux + fx * tau / rho;		// (wzór 4.54)
 		uy = uy + fy * tau / rho;		// ta sama postaÄ dla kierunku y

      	float feq;	   
      	for(int k=0; k< 9; k++)
	    {
         	 // wzór 4.52		
	       feq =  w[ k ] * rho * (1.0f - (3.0f/2.0f) * (ux*ux + uy*uy) + 3.0f * (ex[ k ] * ux + ey[ k ]*uy) 
				+ (9.0f/2.0f) * (ex[ k ] * ux + ey[ k ]*uy) * (ex[ k ] * ux + ey[ k ]*uy));
		  // "f z tyld¹" zapamiętujemy w miejscu aktualnej funkcji rozkładu
	       df [c][ i ][ j ][ k ] =  df [c][ i ][ j ][ k ] - (1/tau)* (df[ c ][ i ][ j ][ k ] - feq);
	    }  	// pętla po kierunkach
	} // pętla po komórkach sieci
 } 

// A6 - krok transportu
void transport(int c)
{
  for(int i=0 ; i < LX ; i++)
  for(int j=0 ; j < LY ; j++)
  if(F[ i ][ j ] == 0)			// tylko z węzła z płynem
  {
    for(int k=0; k< 9; k++)
    {
        int ip = ( i+ex[ k ] + LX ) % (LX);
        int jp = ( j+ey[ k ] + LY ) % (LY); 			// int yp = ( j+ey[ k ] + LY ) % (LY);
         
  		if( F[ip][jp] == 1 )		// docelowy węzeł jest brzegowy?	
	        df[1-c][ i ][ j ][ inv[ k ] ] = df[c][ i ][ j ][ k ];		// tak, wykonaj odbicie
		else
  	        df[1-c][ ip ][ jp ][ k ] = df[c][ i ][ j ][ k ];		// nie, normalny transport
  	 } // pętla po kierunkach
  	} 
 }

void exportvelocity(std::string filename)
{
 ofstream file(filename);
 for(int j=0; j<LY; j++)
    for(int i=0; i<LX; i++)
		file << U[i][j] << " " << V[i][j] << endl;
 file.close();
}

void exportvtk(void)
{
/*
1. # vtk DataFile Version 2.0
2. Komentarz - nasze pole prÄ™dkoÅ›ci policzone metodÄ… LBM
3. ASCII
4. DATASET STRUCTURED_POINTS
5. DIMENSIONS 4 2 1
6. ORIGIN 0 0 0
7. SPACING 1 1 1
8. POINT_DATA 8
10. VECTORS PolePredkosci double
11. 1.0 0.0 0.0*/

	ofstream file("velocity.vtk");
	file << "# vtk DataFile Version 2.0\nLBM, Symulacje Komputerowe w Fizyce 2, Maciej Matyka 2019\n";
	file << "ASCII\nDATASET STRUCTURED_POINTS\n";
	file << "DIMENSIONS " << LX << " " << LY << " 1\n";
	file << "ORIGIN 0 0 0\nSPACING 1 1 1\n";
	file << "POINT_DATA " << LX*LY*1 << "\n";
	file << "VECTORS PolePredkosci double\n";

	for(int j=0; j<LY; j++)
	for(int i=0; i<LX; i++)
	{
		file << U[i][j] << " " << V[i][j] << " 0.0" << endl; 
	}

	file.close();

	// zapisz gestosc-1 (do cisnienia)
	ofstream file2("porosityeff.vtk");
	file2 << "# vtk DataFile Version 2.0\nLBM, Symulacje Komputerowe w Fizyce 2, Maciej Matyka 2019\n";
	file2 << "ASCII\nDATASET STRUCTURED_POINTS\n";
	file2 << "DIMENSIONS " << LX << " " << LY << " 1\n";
	file2 << "ORIGIN 0 0 0\nSPACING 1 1 1\n";
	file2 << "POINT_DATA " << LX*LY*1 << "\n";
	file2 << "SCALARS Gestosc int\n";
	file2 << "LOOKUP_TABLE default\n";
	for(int j=0; j<LY; j++)
	for(int i=0; i<LX; i++)
	{
		if(F_EFF[i][j]==0)
			file2 << 0 << endl; 
		else
			file2 << 1 << endl;
	}
	file2.close();
	// zapisz gestosc-1 (do cisnienia)
/*	ofstream file3("porosity.vtk");
	file3 << "# vtk DataFile Version 2.0\nLBM, Symulacje Komputerowe w Fizyce 2, Maciej Matyka 2019\n";
	file3 << "ASCII\nDATASET STRUCTURED_POINTS\n";
	file3 << "DIMENSIONS " << LX << " " << LY << " 1\n";
	file3 << "ORIGIN 0 0 0\nSPACING 1 1 1\n";
	file3 << "POINT_DATA " << LX*LY*1 << "\n";
	file3 << "SCALARS Gestosc int\n";
	file3 << "LOOKUP_TABLE default\n";
	for(int j=0; j<LY; j++)
	for(int i=0; i<LX; i++)
	{
		if(F[i][j]==0)
			file3 << 0 << endl; 
		else
			file3 << 1 << endl;
	}
	file3.close();*/
}

// A3 - krok LBM
static int lbm_c = 1;

void lbm_reset_parity(void)
{
    lbm_c = 1;
}

void lbm(void)
{
    lbm_c = 1-lbm_c;               // wybierz na której siatce pracujemy (1,0,1,0,1,0...)
    int c = lbm_c;
    macro(c);
    collision(c);
    transport(c);
}


