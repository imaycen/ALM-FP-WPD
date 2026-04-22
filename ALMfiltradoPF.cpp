//*************************************************************************
//
//    Programa que resuelve el modelo para filtrado utilizando punto fijo 
//    y descenso de gradiente para el Metodo de Lagrangiano Aumentado (ALM)
//    realiza K iteraciones de Gauss-Seidel
//
// Author       : Iván de Jesús May-Cen
// Language     : C++
// Compiler     : g++
// Environment  : 
// Revisions
//   Initial    : 2024-03-01 09:36:44 
//   Last       : 
// Se añaden funciones para el calculo de metricas 20 / 02 / 2023
//
//  para compilar
//    g++ -O2 ALMfiltradoPF.cpp -o test -lrt -lblitz `pkg-config --cflags opencv4` `pkg-config --libs opencv4`
//  para ejecutar
//    ./test imgname.png coefR lambda1 lambda2 lambda3
// 
//*************************************************************************

// preprocesor directives
#include <opencv2/core/core.hpp>                // OpenCV      
#include <opencv2/highgui/highgui.hpp>           
#include <blitz/array.h>                                // Blitz++             
#include <random/uniform.h>
#include <random/normal.h>
#include <sys/time.h>                   // funciones de tiempo
#include <cmath>                        // funciones matematicas
#include <float.h>                      // mathematical constants
#include <iostream>                                  
#include <fstream> 

// declara namespace
using namespace std;
using namespace cv;
using namespace blitz;
using namespace ranlib;
// REFERENCIAS
/*
[1] Legarda-Saenz, R., & Brito-Loeza, C. (2020). Augmented Lagrangian method for a total variation-based model for demodulating phase discontinuities. Journal of Algorithms & Computational Technology, 14, 1748302620941413.

[2] May-Cen, I., Legarda-Saenz, R., & Brito-Loeza, C. (2023). A variational model for wrapped phase denoising. Mathematics, 11(12), 2618.
*/



// Especificar dimension de datos sinteticos
//int renglones = 240, columnas = 320;
//int renglones = 480, columnas = 640;
int renglones, columnas;
double *sn0, *cs0; //Variable global para fase ruidosa
double *Q11, *Q12, *Mu11, *Mu12;
double *Q21, *Q22, *Mu21, *Mu22;
double *dxIs, *dyIs, *dxIc, *dyIc;


double *Is0_h, *Ic0_h, *Is1_h, *Ic1_h, *Is2_h, *Ic2_h, *Is11_h, *Ic11_h, *dIs, *dIc;

double TAO = 0.0001, coefR = 1.0, BETA, LAMBDA1, LAMBDA2, LAMBDA3;
char imgname[50];

const double LAMBDA = 1.0;  //para regularizadores TV
const int K = 3;
//const double BETA = 1.0e-3;


// variables del metodo numerico
const double EPSILON1 = 1.0e-6;         // criterio de paro del algoritmo, gradiente
const double EPSILON2 = 1.0e-6;         // criterio de paro del algoritmo, cambio en x
const unsigned ITER_MAX1 = 200000;    // maximo de iteraciones 




// declaracion de funciones
double gradientWrap(double, double);
double minMod(double, double);  
void Print3D(Array<double,2>,FILE*,const char*);
void Print2D(const char*, FILE*, const char*);
void Derivada( double* & derivIs_h, double* & derivIc_h, double* & Is_h, double* & Ic_h );
void solve_descenso_gradiente_ALM( double* &Is_h, double* &Ic_h, double* &derivIs_h, double* &derivIc_h, const char* win1, Array<double, 2> dummy, Mat Imagen );
void solve_gradiente_nesterov_ALM(double* &Is_h, double* &Ic_h, double* &derivIs_h, double* &derivIc_h, const char* win1, Array<double, 2> dummy, Mat Imagen );

// metricas 20/02/2023
double MSE(Array<double,2>& P, Array<double,2>& Po);
double IFI(Array<double,2>& P, Array<double,2>& Po);
double IQI(Array<double,2>& P, Array<double,2>& Po);
double SSI(Array<double,2>& P, Array<double,2>& Po);
double SSMPI(Array<double,2>& P, Array<double,2>& Po);
double NEI(Array<double,2>& P, Array<double,2>& Po);

// Funciones double para calculos numericos
void error_relativo(double* errorRe, double* errorIm, double* &Re, double* &Reo, double* &Im, double* &Imo);
double Funcional( double* &Is_h, double* &Ic_h, double* &Q11, double* &Q12, double* &Q21, double* &Q22, double* &Mu11, double* &Mu12, double* &Mu21, double* &Mu22 );
void iteracion_Gauss_Seidel( double* &Is_h, double* &Ic_h, double*  &Is1_h, double* &Ic1_h);
void punto_Fijo_TV_ALM( double* &Is_h, double* &Ic_h, double* &Is1_h, double* &Ic1_h );
void solve_punto_fijo_ALM(double* &Is_h, double* &Ic_h, double* &Is1_h, double* &Ic1_h, const char* win1, Array<double, 2> dummy, Mat Imagen);
void boundaryCond1( double* &T1, double* &T2, int renglones, int columnas );
void boundaryCondALM1( double* &T1, double* &T2, int renglones, int columnas );
void normas_derivadas(double* errorRe, double* errorIm, double* Re, double* Im);
void actualizaQ( double* &Q11, double* &Q12, double* &Q21, double* &Q22, double* &Mu11, double* &Mu12, double* &Mu21, double* &Mu22, double* &dxIc, double* &dyIc, double* &dxIs, double* &dyIs);
void actualizaMu(double* &Mu11, double* &Mu12, double* &Mu21, double* &Mu22, double* &Q11, double* &Q12, double* &Q21, double* &Q22, double* &dxIc, double* &dyIc, double* &dxIs, double* &dyIs);
void gradiente( double* &Is_h, double* &Ic_h, double* &dxIs_h, double* &dyIs_h, double* &dxIc_h, double* &dyIc_h );
void gradiente2( double* &Is_h, double* &Ic_h, double* &dxIs_h, double* &dyIs_h, double* &dxIc_h, double* &dyIc_h );
//*************************************************************************
//
//                        inicia funcion principal
//
//*************************************************************************
int main( int argc, char **argv )
{
  //parametros desde consola
  if(argc == 6) 
    {
     //version para lectura de imagen
     // read name of the file, read image
     strcpy( imgname, argv[1] );
     coefR = atof(argv[2]);
     LAMBDA1 = atof(argv[3]);
     LAMBDA2 = atof(argv[4]);
     LAMBDA3 = atof(argv[5]);
    }

  // despliega informacion del proceso
  cout << endl << "Inicia procesamiento..." << endl << endl;

  // Leemos datos de la imagen
  Mat IMAGEN = imread(imgname, IMREAD_GRAYSCALE);
  renglones = IMAGEN.rows, columnas = IMAGEN.cols;
     
  // separa memoria para procesar con Blitz   
  Array<double,2> Is(renglones,columnas), Ic(renglones,columnas), phase0(renglones,columnas), dummy(renglones,columnas);
  Array<double,2> Is0(renglones,columnas), Ic0(renglones,columnas), difFase(renglones,columnas), difIs(renglones,columnas), difIc(renglones,columnas);
  Array<double,2> P(renglones,columnas), WP(renglones,columnas), WP0img(renglones,columnas), Is2(renglones,columnas), Ic2(renglones,columnas);

  Array<double,2> Iso(renglones,columnas), Ico(renglones,columnas), derivIs(renglones,columnas), derivIc(renglones,columnas);

  Array<double,2> Is1(renglones,columnas), Ic1(renglones,columnas),Paux(renglones,columnas);
  // crea manejador de imagenes con openCV
  Mat Imagen( renglones, columnas, CV_64F, (unsigned char*) dummy.data() );
  const char *win0 = "Imagen ruidosa";      namedWindow( win0, WINDOW_AUTOSIZE );
  const char *win1 = "Estimaciones";        namedWindow( win1, WINDOW_AUTOSIZE );
  
  // Arreglos para calculos numericos
  double *Is_h, *Ic_h, *derivIs_h, *derivIc_h;
  long int size_matrix = renglones*columnas;
  size_t size_matrix_bytes = size_matrix * sizeof(double);
  Is_h = (double*)malloc(size_matrix_bytes);
  Ic_h = (double*)malloc(size_matrix_bytes);
  dIs = (double*)malloc(size_matrix_bytes);
  dIc = (double*)malloc(size_matrix_bytes);
  Is0_h = (double*)malloc(size_matrix_bytes);
  Ic0_h = (double*)malloc(size_matrix_bytes);  
  Is1_h = (double*)malloc(size_matrix_bytes);
  Ic1_h = (double*)malloc(size_matrix_bytes);  
  Is11_h = (double*)malloc(size_matrix_bytes);
  Ic11_h = (double*)malloc(size_matrix_bytes);  
  Is2_h = (double*)malloc(size_matrix_bytes);
  Ic2_h = (double*)malloc(size_matrix_bytes);  
  derivIs_h = (double*)malloc(size_matrix_bytes);
  derivIc_h = (double*)malloc(size_matrix_bytes);
  sn0 = (double*)malloc(size_matrix_bytes);   
  cs0 = (double*)malloc(size_matrix_bytes); 
  Q11 = (double*)malloc(size_matrix_bytes);
  Q12 = (double*)malloc(size_matrix_bytes);
  Mu11 = (double*)malloc(size_matrix_bytes);
  Mu12 = (double*)malloc(size_matrix_bytes);   
  Q21 = (double*)malloc(size_matrix_bytes);
  Q22 = (double*)malloc(size_matrix_bytes);
  Mu21 = (double*)malloc(size_matrix_bytes);
  Mu22 = (double*)malloc(size_matrix_bytes);  
  dxIs = (double*)malloc(size_matrix_bytes);
  dyIs = (double*)malloc(size_matrix_bytes);
  dxIc = (double*)malloc(size_matrix_bytes);
  dyIc = (double*)malloc(size_matrix_bytes); 
  //lectura de datos sin ruido  
  Mat IMAGENoriginal = imread("phaseEnvuelta.png", IMREAD_GRAYSCALE);  
  
  int tonos = 256; // tonos de gris
  double omega = 2.0 * M_PI / double(tonos);
  double LUT[tonos];
  for ( int x = 0; x < tonos; x++ )
    {
      double fase = omega*double(x+1) - M_PI;
      LUT[x] = atan2( sin(fase), cos(fase) );
    }     
    
  //Lee datos desde imagen
  for ( int r = 0; r < renglones; r++ )
    for ( int c = 0; c < columnas; c++ )
      {
//        // Lee valores desde imagen
        double phase = LUT[IMAGEN.at<unsigned char>(r,c)];//double(IMAGEN.at<uchar>(r,c))/255.0;
        //WP0(r,c) = 2.0 * M_PI * phase - M_PI;

        // para fase inicial Is, Ic con ruido
        Is0(r,c) = sin(phase);//sin(2.0 * M_PI * phase - M_PI);
        Ic0(r,c) = cos(phase);//cos(2.0 * M_PI * phase - M_PI);

        WP0img(r,c) = atan2(Is0(r,c), Ic0(r,c));

        long int idx_r_c = r*columnas + c;        
        sn0[idx_r_c] = Is0(r,c);
        cs0[idx_r_c] = Ic0(r,c);

        // fase envuelta sin ruido para determinar error
        WP(r,c) = LUT[IMAGENoriginal.at<uchar>(r,c)]; 
        ///////WP(r,c) = atan2(sin(phase), cos(phase));
        
        // fase inicial
        Is(r,c) = sn0[idx_r_c];
        Ic(r,c) = cs0[idx_r_c];

        // asignacion a arreglos double
        Is_h[idx_r_c] = Is(r,c);
        Ic_h[idx_r_c] = Ic(r,c);
        Is0_h[idx_r_c] = Is_h[idx_r_c];
        Ic0_h[idx_r_c] = Ic_h[idx_r_c];  

//        // calcula el SNR 
//        num += ( (phase+ruido) * (phase+ruido) );
//        den += ( ruido * ruido );
      }

  // despliega diferencia entre la estimacion y el valor real
  //cout << endl << "SNR = " << 20.0*log10(num/den) << " dB" << endl;

  dummy = (WP0img + M_PI) / (2.0*M_PI);  
  imshow( win0, Imagen );

  // guarda imagenes iniciales
  dummy = (WP + M_PI) / (2.0*M_PI); 
  imwrite( "imagenes/phaseEnvuelta.png", 255*Imagen );  
  dummy = (WP0img + M_PI) / (2.0*M_PI); 
  imwrite( "imagenes/phaseEnvueltaRuidosa-AN.png", 255*Imagen );
  dummy = Ic;
  normalize( Imagen, Imagen, 0, 1, NORM_MINMAX );
  imwrite( "imagenes/IcRuidosa-AN.png", 255*Imagen );
  dummy = Is;
  normalize( Imagen, Imagen, 0, 1, NORM_MINMAX );
  imwrite( "imagenes/IsRuidosa-AN.png", 255*Imagen );

  // ************************************************************************
  //             Inicia procesamiento
  // ************************************************************************
  double start_time, end_time;
  start_time = (double)cv::getTickCount();

  
  // Elije metodo de solucion para AML-WPD
  // 1 : Gradient descent
  // 2 : Nesterov acelerated gradient
  // 3 : Fixed-Poin
  int metodo = 3;
    switch ( metodo )
    {
      case 1:      // Descenso de gradiente
        solve_descenso_gradiente_ALM(Is_h, Ic_h, derivIs_h, derivIc_h, win1, dummy, Imagen);
        break;
      case 2:      // NAG
        solve_gradiente_nesterov_ALM(Is_h, Ic_h, derivIs_h, derivIc_h, win1, dummy, Imagen);
        break;   
      case 3:      // Fixed-point
        solve_punto_fijo_ALM(Is_h, Ic_h, Is1_h, Ic1_h, win1, dummy, Imagen);
        break;     
     }

  // termina funcion, calcula y despliega valores indicadores del proceso  
  end_time = (double)cv::getTickCount(); 
  cout << endl << "Tiempo empleado : " << (end_time - start_time) / cv::getTickFrequency() << endl; 

  // ************************************************************************
  //   resultados del procesamiento
  // ************************************************************************

   
   //genera imagen resultante 
   for(int r = 0; r < renglones; r++)
     for(int c = 0; c < columnas; c++)
      {
       long int idx_r_c = r*columnas + c;
       Is(r,c) = Is_h[idx_r_c];
       Ic(r,c) = Ic_h[idx_r_c];
       //para grafica de identidad pitagorica
       double sn = Is(r,c);
       double cs = Ic(r,c);
       P(r,c) = abs(sn*sn + cs*cs - 1.0);
      }
  
// calculo de errores finales
  WP0img = atan2(Is, Ic);
  Is0 = sin(WP);
  Ic0 = cos(WP);
  
  cout << "Errores en la parte real" << endl;
  cout << "MSE : " << MSE(Ic0, Ic) << endl; 
  cout << "NEI : " << NEI(Ic0, Ic) << endl; 
  cout << "IFI : " << IFI(Ic0, Ic) << endl;   
//  cout << "IQI : " << IQI(Ic0, Ic) << endl;  
//  cout << "SSI : " << SSI(Ic0, Ic) << endl; 
//  cout << "SSMPI : " << SSMPI(Ic0, Ic) << endl;   
  
  cout << "Errores en la parte imaginaria" << endl;
  cout << "MSE : " << MSE(Is0, Is) << endl; 
  cout << "NEI : " << NEI(Is0, Is) << endl; 
  cout << "IFI : " << IFI(Is0, Is) << endl; 
//  cout << "IQI : " << IQI(Is0, Is) << endl; 
//  cout << "SSI : " << SSI(Is0, Is) << endl; 
//  cout << "SSMPI : " << SSMPI(Is0, Is) << endl;
  
  cout << "Errores en la fase envuelta" << endl;
  cout << "MSE : " << MSE(WP, WP0img) << endl; 
  cout << "NEI : " << NEI(WP, WP0img) << endl;   
  cout << "IFI : " << IFI(WP, WP0img) << endl; 
//  cout << "IQI : " << IQI(WP, WP0img) << endl; 
//  cout << "SSI : " << SSI(WP, WP0img) << endl; 
//  cout << "SSMPI : " << SSMPI(WP, WP0img) << endl;
  
  
  
  
  // guarda en archivo la estimacion
  dummy = Is;
  normalize( Imagen, Imagen, 0, 1, NORM_MINMAX );
  imwrite( "imagenes/IsEstimada-ALM-AN.png", 255*Imagen );
  dummy = Ic;
  normalize( Imagen, Imagen, 0, 1, NORM_MINMAX );
  imwrite( "imagenes/IcEstimada-ALM-AN.png", 255*Imagen );
  dummy = (atan2( Is, Ic ) + M_PI) / (2.0*M_PI);
  imwrite( "imagenes/phaseEstimada-ALM-AN.png", 255*Imagen );

//// calculo de graficas de errores

//  difIc = abs(Ic-Ic0);
//  difIs = abs(Is-Is0);
//  difFase = abs(WP0img-WP);
//    
//  //normalize( Imagen, Imagen, 0, 1, NORM_MINMAX );    
//  dummy = difIc;//(difIc + M_PI) / (2.0*M_PI);
//  //normalize( Imagen, Imagen, 0, 1, NORM_MINMAX ); 
//  imwrite( "imagenes/difIc-ANMN.png", 255*Imagen );
//  dummy = difIs;//(difIs + M_PI) / (2.0*M_PI);
//  //normalize( Imagen, Imagen, 0, 1, NORM_MINMAX ); 
//  imwrite( "imagenes/difIs-ANMN.png", 255*Imagen );
//  dummy = difFase;//(difFase + M_PI) / (2.0*M_PI);
//  imwrite( "imagenes/difFase-ANMN.png", 255*Imagen );


  // Grafica resultados de errores en 3D   
//  FILE* gnuplot_pipe1 = popen( "gnuplot -p", "w" );
//  Print3D( difIc, gnuplot_pipe1, "imagenes/difIc-ANMN.eps" );
//  pclose( gnuplot_pipe1 );     
//  
//  FILE* gnuplot_pipe2 = popen( "gnuplot -p", "w" );
//  Print3D( difIs, gnuplot_pipe1, "imagenes/difIs-ANMN.eps" );
//  pclose( gnuplot_pipe2 );  
//  
//  FILE* gnuplot_pipe3 = popen( "gnuplot -p", "w" );
//  Print3D( difFase, gnuplot_pipe3, "imagenes/difFase-ANMN.eps" );
//  pclose( gnuplot_pipe3 ); 
  
  FILE* gnuplot_pipe4 = popen( "gnuplot -p", "w" );
  Print3D( P, gnuplot_pipe4, "imagenes/identidad-ALM-AN.eps" );
  pclose( gnuplot_pipe4 );   
  
  // termina ejecucion del programa



  free(Is_h);
  free(Ic_h);
  free(dIs);
  free(dIc);
  free(Is0_h);
  free(Ic0_h);
  free(Is1_h);
  free(Ic1_h);
  free(Is2_h);
  free(Ic2_h);
  free(Is11_h);
  free(Ic11_h);
  free(derivIs_h);
  free(derivIc_h);
  free(sn0);
  free(cs0);
  free(Q11);
  free(Q12);
  free(Mu11);
  free(Mu12);
  free(Q21);
  free(Q22);
  free(Mu21);
  free(Mu22);
  free(dxIs);
  free(dyIs);
  free(dxIc);
  free(dyIc);
  // termina ejecucion del programa
  return 0;
}
//*************************************************************************
//
//    Funciones de trabajo
//
//*************************************************************************
// ************************************************************************
//       funcion para Gauss-Seidel para double
//              Punto-fijo para ALM
//*************************************************************************
void punto_Fijo_TV_ALM( double* &Is_h, double* &Ic_h, double* &Is1_h, double* &Ic1_h )
{
  long int SizeImage = renglones*columnas;
  
  //condiciones de frontera Neumann para Is, Ic
  boundaryCond1( Ic_h, Is_h, renglones, columnas );
  for(long int idx_r_c = 0; idx_r_c < SizeImage; idx_r_c++) 
     {
      Is1_h[idx_r_c] = Is_h[idx_r_c];
      Ic1_h[idx_r_c] = Ic_h[idx_r_c];
     }
  // calculo de punto fijo K iteraciones de GS
  for ( int k = 0; k < K; k++ )
  {
    iteracion_Gauss_Seidel(Ic_h, Is_h, Ic1_h, Is1_h);

     for(long int idx_r_c = 0; idx_r_c < SizeImage; idx_r_c++)
       {
        Is_h[idx_r_c] = Is1_h[idx_r_c];
        Ic_h[idx_r_c] = Ic1_h[idx_r_c];
       }
    }
  
}
void iteracion_Gauss_Seidel(double* &Ic_h, double* &Is_h, double* &Ic1_h, double* &Is1_h)
{
  // define parametro de regularizacion
  double lambda1 = LAMBDA1, lambda2 = LAMBDA2, lambda3 = LAMBDA3;//, lambda4 = LAMBDA4, lambda5 = LAMBDA5;
  //double beta = BETA;
  double V1x, V2x, V1y, V2y, Ux, Uy;
  double dxMu11, dyMu12, divMu1, dxMu21, dyMu22, divMu2, numIs, denIs;
  double dxQ11, dyQ12, divQ1, dxQ21, dyQ22, divQ2, numIc, denIc;
  double residuo, aux;
  
  for ( long int r = 1; r < renglones-1; r++ )
    for ( long int c = 1; c < columnas-1; c++ )
      {
       long int idx_r_c = r*columnas + c;
       long int idx_rp1_c = (r + 1)*columnas + c;
       long int idx_rm1_c = (r - 1)*columnas + c;
       long int idx_r_cp1 = r*columnas + c + 1;
       long int idx_r_cm1 = r*columnas + c - 1; 
       long int idx_rm1_cp1 = (r - 1)*columnas + c + 1; 
       long int idx_rp1_cm1 = (r + 1)*columnas + c - 1; 

//           Se obtienen derivadas parciales 
//           para calculo de divergencia
//           long int idx_r_c = r*columnas + c;
//           long int idx_r_cp1 = r*columnas + c + 1;
       dyMu12 = Mu12[idx_r_cp1]-Mu12[idx_r_c];
       dyMu22 = Mu22[idx_r_cp1]-Mu22[idx_r_c];
       dyQ12 = Q12[idx_r_cp1]-Q12[idx_r_c];
       dyQ22 = Q22[idx_r_cp1]-Q22[idx_r_c];           

//           long int idx_r_c = r*columnas + c;
//           long int idx_r_cm1 = r*columnas + c - 1; 
       dyMu12 = Mu12[idx_r_c]-Mu12[idx_r_cm1];
       dyMu22 = Mu22[idx_r_c]-Mu22[idx_r_cm1];
       dyQ12 = Q12[idx_r_c]-Q12[idx_r_cm1];
       dyQ22 = Q22[idx_r_c]-Q22[idx_r_cm1];

//         long int idx_r_c = r*columnas + c;
//     	   long int idx_r_cp1 = r*columnas + c + 1; 
       dyMu12 = Mu12[idx_r_cp1]-Mu12[idx_r_c];
       dyMu22 = Mu22[idx_r_cp1]-Mu22[idx_r_c];
       dyQ12 = Q12[idx_r_cp1]-Q12[idx_r_c];
       dyQ22 = Q22[idx_r_cp1]-Q22[idx_r_c];

//         long int idx_r_c = r*columnas + c;
//	   long int idx_rp1_c = (r + 1)*columnas + c;
       dxMu11 = Mu11[idx_rp1_c]-Mu11[idx_r_c];
       dxMu21 = Mu21[idx_rp1_c]-Mu21[idx_r_c];
       dxQ11 = Q11[idx_rp1_c]-Q11[idx_r_c];
       dxQ21 = Q21[idx_rp1_c]-Q21[idx_r_c];
 
//           long int idx_r_c = r*columnas + c;
//           long int idx_rm1_c = (r - 1)*columnas + c;
        dxMu11 = Mu11[idx_r_c]-Mu11[idx_rm1_c];
        dxMu21 = Mu21[idx_r_c]-Mu21[idx_rm1_c];    
        dxQ11 = Q11[idx_r_c]-Q11[idx_rm1_c];
        dxQ21 = Q21[idx_r_c]-Q21[idx_rm1_c];        

//           long int idx_r_c = r*columnas + c;
//           long int idx_rp1_c = (r + 1)*columnas + c;
        dxMu11 = Mu11[idx_rp1_c]-Mu11[idx_r_c]; 
        dxMu21 = Mu21[idx_rp1_c]-Mu21[idx_r_c];
        dxQ11 = Q11[idx_rp1_c]-Q11[idx_r_c]; 
        dxQ21 = Q21[idx_rp1_c]-Q21[idx_r_c];
       // termina calculo de derivadas parciales
   
       //Obtener div Mu1^k y div Mu2^k       
       divMu1 = -1.0*(dxMu11 + dyMu12);
       divMu2 = -1.0*(dxMu21 + dyMu22);
       
       //Obtener div Q1^k y div Q2^k
       divQ1 = -1.0*(dxQ11 + dyQ12);
       divQ2 = -1.0*(dxQ21 + dyQ22);

        // iteracion de Gauss-Seidel
        numIs = lambda2*sn0[idx_r_c] + 2.0*lambda3*Is_h[idx_r_c] + coefR*(Is_h[idx_rp1_c] + Is1_h[idx_rm1_c] + Is_h[idx_r_cp1] + Is1_h[idx_r_cm1]) - divMu2 - coefR*divQ2;

        numIc = lambda1*cs0[idx_r_c] + 2.0*lambda3*Ic_h[idx_r_c] + coefR*(Ic_h[idx_rp1_c] + Ic1_h[idx_rm1_c] + Ic_h[idx_r_cp1] + Ic1_h[idx_r_cm1]) - divMu1 - coefR*divQ1; 

        aux = 2.0*lambda3*( Is_h[idx_r_c]*Is_h[idx_r_c] + Ic_h[idx_r_c]*Ic_h[idx_r_c] );

        denIs = lambda2 + aux + 4.0*coefR;
        denIc = lambda1 + aux + 4.0*coefR;
        Is1_h[idx_r_c] = numIs / denIs;
        Ic1_h[idx_r_c] = numIc / denIc;
      }
       //actualiza gradientes para calculo de Q y Mu
       gradiente( Is1_h, Ic1_h, dxIs, dyIs, dxIc, dyIc );

       //soft-thresholding operator
       //Actualizar Q1^k
       //Actualizar Q2^k
       actualizaQ(Q11, Q12, Q21, Q22, Mu11, Mu12, Mu21, Mu22, dxIc, dyIc, dxIs, dyIs);      
       //Actualizar Mu1^k
       //Actualizar Mu2^k     
       actualizaMu(Mu11, Mu12, Mu21, Mu22, Q11, Q12, Q21, Q22, dxIc, dyIc, dxIs, dyIs);

}
// ***************************************************************
//   Condiciones de frontera Neumann para double*, double*
// ***************************************************************
void boundaryCond1( double* &T1, double* &T2, int renglones, int columnas )
{
	// condiciones de frontera
	//T(0, all) = T(1, all);
	//T(renglones - 1, all) = T(renglones - 2, all);
//#pragma omp parallel for firstprivate(renglones,columnas) num_threads(N_threads)
	for (int c = 0; c < columnas; c++){
		long int idx_0_c = c;
		long int idx_1_c = columnas + c;
		long int idx_rm1_c = (renglones - 1)*columnas + c;
		long int idx_rm2_c = (renglones - 2)*columnas + c;

		T1[idx_0_c] = T1[idx_1_c];
		T1[idx_rm1_c] = T1[idx_rm2_c];
		T2[idx_0_c] = T2[idx_1_c];
		T2[idx_rm1_c] = T2[idx_rm2_c];
	}
	//T(all, 0) = T(all, 1);
	//T(all, columnas - 1) = T(all, columnas - 2);
//#pragma omp parallel for firstprivate(renglones,columnas) num_threads(N_threads)
	for (int r = 0; r < renglones; r++){
		long int idx_r_0 = r*columnas;
		long int idx_r_1 = r*columnas + 1;
		long int idx_r_cm1 = r*columnas + columnas - 1;
		long int idx_r_cm2 = r*columnas + columnas - 2;

		T1[idx_r_0] = T1[idx_r_1];
		T1[idx_r_cm1] = T1[idx_r_cm2];
		T2[idx_r_0] = T2[idx_r_1];
		T2[idx_r_cm1] = T2[idx_r_cm2];
	}

	//T(0, 0) = T(1, 1);
	T1[0] = T1[columnas + 1];
	//T(0, columnas - 1) = T(1, columnas - 2);
	T1[columnas - 1] = T1[columnas + columnas - 2];
	//T(renglones - 1, 0) = T(renglones - 2, 1);
	T1[(renglones - 1)*columnas] = T1[(renglones - 2)*columnas + 1];
	//T(renglones - 1, columnas - 1) = T(renglones - 2, columnas - 2);
	T1[(renglones - 1)*columnas + columnas - 1] = T1[(renglones - 2)*columnas + columnas - 2];
	
	T2[0] = T2[columnas + 1];
	T2[columnas - 1] = T2[columnas + columnas - 2];
	T2[(renglones - 1)*columnas] = T2[(renglones - 2)*columnas + 1];
	T2[(renglones - 1)*columnas + columnas - 1] = T2[(renglones - 2)*columnas + columnas - 2];	
	
}
// ***************************************************************
//   Condiciones de frontera para problema ALM para double*, double*
// ***************************************************************
void boundaryCondALM1( double* &T1, double* &T2, int renglones, int columnas )
{
	// condiciones de frontera
	//T(0, all) = T(1, all);
	//T(renglones - 1, all) = T(renglones - 2, all);
//#pragma omp parallel for firstprivate(renglones,columnas) num_threads(N_threads)
	for (long int c = 0; c < columnas; c++){
		long int idx_0_c = c;
		long int idx_1_c = columnas + c;
		long int idx_rm1_c = (renglones - 1)*columnas + c;
		long int idx_rm2_c = (renglones - 2)*columnas + c;
		
		Mu11[idx_0_c] = 0.0;
	        Mu21[idx_0_c] = 0.0;
	        Mu11[idx_rm1_c] = 0.0;
	        Mu21[idx_rm1_c] = 0.0;
	        Mu12[idx_0_c] = 0.0;
	        Mu22[idx_0_c] = 0.0;
	        Mu12[idx_rm1_c] = 0.0;
	        Mu22[idx_rm1_c] = 0.0;

		T1[idx_0_c] = T1[idx_1_c] - Q11[idx_0_c];
		T1[idx_rm1_c] = T1[idx_rm2_c] + Q11[idx_rm2_c];
		
		T2[idx_0_c] = T2[idx_1_c] - Q21[idx_0_c];
		T2[idx_rm1_c] = T2[idx_rm2_c] + Q21[idx_rm2_c];
	}
	//T(all, 0) = T(all, 1);
	//T(all, columnas - 1) = T(all, columnas - 2);
//#pragma omp parallel for firstprivate(renglones,columnas) num_threads(N_threads)
	for (long int r = 0; r < renglones; r++){
		long int idx_r_0 = r*columnas;
		long int idx_r_1 = r*columnas + 1;
		long int idx_r_cm1 = r*columnas + columnas - 1;
		long int idx_r_cm2 = r*columnas + columnas - 2;
		
		Mu11[idx_r_0] = 0.0;
	        Mu21[idx_r_0] = 0.0;
	        Mu11[idx_r_cm1] = 0.0;
	        Mu21[idx_r_cm1] = 0.0;
	        Mu12[idx_r_0] = 0.0;
	        Mu22[idx_r_0] = 0.0;
	        Mu12[idx_r_cm1] = 0.0;
	        Mu22[idx_r_cm1] = 0.0;

		T1[idx_r_0] = T1[idx_r_1] - Q12[idx_r_0];
		T1[idx_r_cm1] = T1[idx_r_cm2] + Q12[idx_r_cm2];
		
		T2[idx_r_0] = T2[idx_r_1] - Q22[idx_r_0];
		T2[idx_r_cm1] = T2[idx_r_cm2] + Q22[idx_r_cm2];
	}

	//T(0, 0) = T(1, 1);
	T1[0] = T1[columnas + 1] - Q11[0];
	//T(0, columnas - 1) = T(1, columnas - 2);
	T1[columnas - 1] = T1[columnas + columnas - 2] + Q12[columnas + columnas - 2];
	//T(renglones - 1, 0) = T(renglones - 2, 1);
	T1[(renglones - 1)*columnas] = T1[(renglones - 2)*columnas + 1] + Q11[(renglones - 2)*columnas + 1];
	//T(renglones - 1, columnas - 1) = T(renglones - 2, columnas - 2);
	T1[(renglones - 1)*columnas + columnas - 1] = T1[(renglones - 2)*columnas + columnas - 2] + Q12[(renglones - 2)*columnas + columnas - 2];
	
	T2[0] = T2[columnas + 1] - Q21[0];
	T2[columnas - 1] = T2[columnas + columnas - 2] + Q22[columnas + columnas - 2];
	T2[(renglones - 1)*columnas] = T2[(renglones - 2)*columnas + 1] + Q21[(renglones - 2)*columnas + 1];
	T2[(renglones - 1)*columnas + columnas - 1] = T2[(renglones - 2)*columnas + columnas - 2] + Q22[(renglones - 2)*columnas + columnas - 2];	
	
}

// ***************************************************************
//   min-mod
// ***************************************************************
double minMod( double a, double b )
{
  // minmod operator
  double signa = (a > 0.0) ? 1.0 : ((a < 0.0) ? -1.0 : 0.0);
  double signb = (b > 0.0) ? 1.0 : ((b < 0.0) ? -1.0 : 0.0);
//  double minim = fmin( fabs(a), fabs(b) ); 
  double minim = ( fabs(a) <= fabs(b) ) ? fabs(a) : fabs(b); 
  return ( (signa+signb)*minim/2.0 );

  // geometric average
//  return( 0.5*(a+b) ); Total Variation Diminishing Runge-Kutta Schemes
  
  // upwind 
//  double maxa = (a > 0.0) ? a : 0.0;
//  double maxb = (b > 0.0) ? b : 0.0;
//  return( 0.5*(maxa+maxb) );  
}
// ************************************************************************
//       funcional para double
//*************************************************************************
double Funcional( double* &Is_h, double* &Ic_h, double* &Q11, double* &Q12, double* &Q21, double* &Q22, double* &Mu11, double* &Mu12, double* &Mu21, double* &Mu22 )
{
  // define parametro de regularizacion
  double lambda1 = LAMBDA1, lambda2 = LAMBDA2, lambda3 = LAMBDA3, val = 0.0, v0, v1, v2, v3, v4, v5, v6, v7, a1, a2, a3;
  double t1, t2;
  double difQIc1, difQIc2, difQIs1, difQIs2;
  double dxIs, dyIs, dxIc, dyIc;
  double hx = 1.0 / (double(renglones)-1.0);
  double hy = 1.0 / (double(columnas)-1.0);

  // evalua derivadas parciales para funcional
  for ( long int r = 0; r < renglones; r++ )
    for ( long int c = 0; c < columnas; c++ )
      {
        // campo de gradiente de la informacion
        if ( c == 0 )
          {  
           long int idx_r_c = r*columnas + c;
           long int idx_r_cp1 = r*columnas + c + 1;
           dyIs = Is_h[idx_r_cp1]-Is_h[idx_r_c];//Is(r,c+1) - Is(r,c);
           dyIc = Ic_h[idx_r_cp1]-Ic_h[idx_r_c];//Ic(r,c+1) - Ic(r,c); 
           }
        else if ( c == columnas-1 )
          {  
           long int idx_r_c = r*columnas + c;
           long int idx_r_cm1 = r*columnas + c - 1; 
           dyIs = Is_h[idx_r_c]-Is_h[idx_r_cm1];//Is(r,c)-Is(r,c-1);
           dyIc = Ic_h[idx_r_c]-Ic_h[idx_r_cm1];//Ic(r,c)-Ic(r,c-1); 
           }
        else
          {  
           long int idx_r_c = r*columnas + c;
     	   long int idx_r_cp1 = r*columnas + c + 1; 
           dyIs = Is_h[idx_r_cp1]-Is_h[idx_r_c];//Is(r,c+1) - Is(r,c);
           dyIc = Ic_h[idx_r_cp1]-Ic_h[idx_r_c];//Ic(r,c+1) - Ic(r,c);
          }
          
        // campo de gradiente de la informacion
        if ( r == 0 )
          {  
           long int idx_r_c = r*columnas + c;
	   long int idx_rp1_c = (r + 1)*columnas + c;
           dxIs = Is_h[idx_rp1_c]-Is_h[idx_r_c];
           dxIc = Ic_h[idx_rp1_c]-Ic_h[idx_r_c];
          }
        else if ( r == renglones-1 )
          {  
           long int idx_r_c = r*columnas + c;
           long int idx_rm1_c = (r - 1)*columnas + c;
           dxIs = Is_h[idx_r_c]-Is_h[idx_rm1_c];
           dxIc = Ic_h[idx_r_c]-Ic_h[idx_rm1_c];
          }
        else
          {  
           long int idx_r_c = r*columnas + c;
           long int idx_rp1_c = (r + 1)*columnas + c;
           dxIs = Is_h[idx_rp1_c]-Is_h[idx_r_c]; 
           dxIc = Ic_h[idx_rp1_c]-Ic_h[idx_r_c];
           }
  
   
       // termina calculo de derivadas parciales de fase
       long int idx_r_c = r*columnas + c;
       a1 = Is_h[idx_r_c]-sn0[idx_r_c];
       a2 = Ic_h[idx_r_c]-cs0[idx_r_c];
       a3 = Is_h[idx_r_c]*Is_h[idx_r_c] + Ic_h[idx_r_c]*Ic_h[idx_r_c] - 1.0;
//       double sn0b = 0.5*(sn0[idx_r_c]+1.0);
//       double cs0b = 0.5*(cs0[idx_r_c]+1.0);
       //t1 = 0.5*Is_h[idx_r_c]+0.5 + (sn0b)*(exp(-0.5*Is_h[idx_r_c]-0.5));
       //t2 = 0.5*Ic_h[idx_r_c]+0.5 + (cs0b)*(exp(-0.5*Ic_h[idx_r_c]-0.5));
//       t1 = Is_h[idx_r_c] + (sn0b)*(exp(-Is_h[idx_r_c]));
//       t2 = Ic_h[idx_r_c] + (cs0b)*(exp(-Ic_h[idx_r_c]));       
       v0 = 0.5 * lambda3 * a3 * a3;
       v1 = 0.5*lambda2*a1*a1 + 0.5*lambda1*a2*a2;
//       v2 = lambda*sqrt(dxIs*dxIs + dyIs*dyIs);
//       v3 = lambda*sqrt(dxIc*dxIc + dyIc*dyIc);
//       v4 = lambda5*t1 + lambda4*t2;
       v2 = sqrt(Q11[idx_r_c]*Q11[idx_r_c] + Q12[idx_r_c]*Q12[idx_r_c]);
       v3 = sqrt(Q21[idx_r_c]*Q21[idx_r_c] + Q12[idx_r_c]*Q12[idx_r_c]);
       difQIc1 = Q11[idx_r_c]-dxIc;
       difQIc2 = Q12[idx_r_c]-dyIc;
       v4 = Mu11[idx_r_c]*difQIc1 + Mu12[idx_r_c]*difQIc2;
       difQIs1 = Q21[idx_r_c]-dxIs;
       difQIs2 = Q22[idx_r_c]-dyIs;
       v5 = Mu21[idx_r_c]*difQIs1 + Mu22[idx_r_c]*difQIs2;
       v6 = 0.5*coefR*(difQIc1*difQIc1 + difQIc2*difQIc2);   
       v7 = 0.5*coefR*(difQIs1*difQIs1 + difQIs2*difQIs2);                 
       val += v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
      }

  // regresa valor
  return val * hx * hy;
}

//*************************************************************************
//      obtiene las diferencias envueltas del termino de fase
//*************************************************************************
double gradientWrap( double p1, double p2 )
{
  double r = p1 - p2;
  return atan2( sin(r), cos(r) ); 
}

//*************************************************************************
//
//    Funciones para despliegue de resultados
//
//*************************************************************************
void Print3D( Array<double,2> Z, FILE* salida, const char* fileName )
{
  // salida en modo grafico o en archivo
  fprintf( salida, "set terminal postscript eps enhanced rounded\n");
  fprintf( salida, "set output \"%s\"\n", fileName );  
  fprintf( salida, "set terminal postscript eps 20\n"); //tamaño de letra eps 20
  fprintf( salida, "set style line 1 linetype -1 linewidth 1\n" );
  fprintf( salida, "set xlabel \"Y\" offset -1,-1\n" );
  fprintf( salida, "set ylabel \"X\" offset -2,-2\n" );
  fprintf( salida, "set zlabel \"Z\" offset 1, 0\n" );
  fprintf( salida, "set xrange [%f:%f]\n", 0., float(Z.cols()) );
  fprintf( salida, "set yrange [%f:%f]\n", 0., float(Z.rows()) );
  //fprintf( salida, "set zrange [%f:%f]\n", -100.0, 100.0 );
  fprintf( salida, "set zrange [%f:%f]\n", 0.0, 1.0 );
  fprintf( salida, "set xtics 100 offset -0.5,-0.5\n" );
  fprintf( salida, "set ytics 100 offset -0.5,-0.5\n" );
  fprintf( salida, "set view 70, 210\n" );
  fprintf( salida, "unset key\n" );
  fprintf( salida, "unset colorbox\n" );
  fprintf( salida, "set hidden3d front\n" );
//  fprintf( salida, "set xyplane at -45.0\n" );
  fprintf( salida, "splot '-' using 1:2:3 title '' with lines lt -1 lw 0.1\n" );
  for ( int c = 0; c < Z.cols(); c += 8 )
    {
      for ( int r = 0; r < Z.rows(); r += 8 )
        fprintf( salida, "%f %f %f\n", float(c), float(r), float(Z(r,c)) );
      fprintf(  salida, "\n" );      // New row (datablock) separated by blank record
    }
  fprintf( salida, "e\n" );
  fflush( salida );
}
//*************************************************************************
void Print2D( const char* filedata, FILE* salida, const char* fileName )
{
  // salida en modo grafico o en archivo
  fprintf( salida, "set terminal postscript eps enhanced rounded\n");
  fprintf( salida, "set output \"%s\"\n", fileName );  
  fprintf( salida, "set style line 1 linetype -1 linewidth 1\n" );
  fprintf( salida, "set xlabel \"iteraciones\" offset -1,-1\n" );
  fprintf( salida, "set ylabel \"gradiente\" offset -1,-1\n" );
  //fprintf( salida, "set title \" %s \" tc lt 1\n");
//  fprintf( salida, "set xrange [%f:%f]\n", 0., float(Z.cols()) );
//  fprintf( salida, "set yrange [%f:%f]\n", 0., float(Z.rows()) );
  fprintf( salida, "set xtics 100 offset -0.5,-0.5\n" );
  fprintf( salida, "set ytics 100 offset -0.5,-0.5\n" );
  fprintf( salida, "plot \"%s\"\n", filedata);
  fprintf( salida, "exit\n" );
  fflush( salida );
}
//***************************************************
// Error relativo para parte real y parte imaginaria
// P = nueva estimacion
// Po = estimacion anterior
//***************************************************
void error_relativo(double* errorRe, double* errorIm, double* &Re, double* &Reo, double* &Im, double* &Imo)
{
double sum_pow2difRRo = 0.0, sum_pow2Reo = 0.0, sum_pow2difIIo = 0.0, sum_pow2Imo = 0.0;
long int SizeImage = renglones * columnas;


for (long int idx_r_c = 0; idx_r_c < SizeImage; idx_r_c++) 
  {
   double vRe = Re[idx_r_c];
   double vReo = Reo[idx_r_c];
   double difRe = vRe-vReo;
   sum_pow2difRRo += difRe*difRe;
   sum_pow2Reo += vReo*vReo;  
   double vIm = Im[idx_r_c];
   double vImo = Imo[idx_r_c];
   double difIm = vIm-vImo;
   sum_pow2difIIo += difIm*difIm;
   sum_pow2Imo += vImo*vImo;     
  }

*errorRe = sqrt(sum_pow2difRRo) / sqrt(sum_pow2Reo);
*errorIm = sqrt(sum_pow2difIIo) / sqrt(sum_pow2Imo);
}
//***************************************************
// Normas de las derivadas para parte real y parte imaginaria
// Re = arreglo para derivada parcial
// Im = arreglo para derivada parcial
//***************************************************
void normas_derivadas(double* errorRe, double* errorIm, double* Re, double* Im)
{
double sum_pow2difRRo = 0.0, sum_pow2difIIo = 0.0;
long int SizeImage = renglones * columnas;


for (long int idx_r_c = 0; idx_r_c < SizeImage; idx_r_c++) 
  {
   double vRe = Re[idx_r_c];
   sum_pow2difRRo += vRe*vRe;
   double vIm = Im[idx_r_c];
   sum_pow2difIIo += vIm*vIm;     
  }

*errorRe = sqrt(sum_pow2difRRo);
*errorIm = sqrt(sum_pow2difIIo);
}
//***************************************************
// Funciones para calculo de metricas
// Añadido 20 / 02 / 2023
//***************************************************
//***************************************************
// 
//***************************************************
double MSE(Array<double,2>& P, Array<double,2>& Po)
{
  int columnas = P.cols();
  int renglones = P.rows();
  
  double mse = sum(pow2(P-Po)) / (double(renglones) * double(columnas));

return mse;
}
//***************************************************
// Image Fidelity Index
// P = datos sin ruido
// Po = estimacion
//***************************************************
double IFI(Array<double,2>& P, Array<double,2>& Po)
{
  
  double iqi = 1.0 - ( sum(pow2(P-Po)) / sum(pow2(P)) );

return iqi;
}
//***************************************************
// Image Quality Index
//
// Calcula el indice de calidad de imagen universal [Wang02] que 
// se puede utilizar como medida de distorsion de calidad de 
// imagen y video. Se define matematicamente modelando  
// la distorsion de la imagen relativa a la imagen de referencia 
// como una combinacion de tres factores: perdida de correlacion,
// distorsion de luminancia y distorsion de contraste.
//
// Fuente : Wang, Z., & Bovik, A. C. (2002). A universal image quality index. IEEE signal processing letters, 9(3), 81-84.
// Fuente : https://ww2.lacan.upc.edu/doc/intel/ipp/ipp_manual/IPPI/ippi_ch11/ch11_image_quality_index.htm
// P = datos sin ruido
// Po = estimacion
//***************************************************
double IQI(Array<double,2>& P, Array<double,2>& Po)
{
  long int SizeImage = P.rows()*P.cols();
  double constante = 1.0 / (SizeImage - 1.0);
  double meanP = sum(P) * (1.0 / SizeImage );
  double meanPo = sum(Po) * (1.0 / SizeImage );
  double sum_sigmaPPo = 0.0, sum_sigma2P = 0.0, sum_sigma2Po = 0.0;
  double sigmaPPo, sigma2P, sigma2Po;
  double q1, q2, q3;
  
  for(int r = 0; r < P.rows(); r++) 
    for(int c = 0; c < P.cols(); c++) 
       {
        double difP = P(r,c)-meanP;
        double difPo = Po(r,c)-meanPo;
        
        sum_sigmaPPo += difP*difPo;
        sum_sigma2P += difP*difP;
        sum_sigma2Po += difPo*difPo;    
       }
  sigmaPPo = constante * sum_sigmaPPo;
  sigma2P = constante * sum_sigma2P;
  sigma2Po = constante * sum_sigma2Po;  
  
  //coeficiente de correlacion mide relacion de linealidad [-1, 1]
  q1 = sigmaPPo / (sqrt(sigma2P) * sqrt(sigma2Po));
  
  //mide que tan cerca esta la luminancia media entre imagenes [0, 1]
  q2 = (2.0 * meanP * meanPo) / ( meanP*meanP + meanPo*meanPo );
  
  //mide la similitud de los contrastes de las imagenes [0, 1]
  q3 = (2.0 * sqrt(sigma2P) * sqrt(sigma2Po)) / (sigma2P + sigma2Po);
 
  cout << "q1 : " << q1 << " q2 : " << q2 << " q3 : " << q3 << endl; 
  double q = q1*q2*q3;

return q;
}
//********************************************************
// Speckle Suppression Index (SSI)
//
// P = datos sin ruido
// Po = estimacion
//********************************************************
double SSI(Array<double,2>& P, Array<double,2>& Po)
{
  long int SizeImage = P.rows()*P.cols();
  double constante = 1.0 / (SizeImage - 1.0);
  double meanP = sum(P) * (1.0 / SizeImage );
  double meanPo = sum(Po) * (1.0 / SizeImage );
  double sum_sigma2P = 0.0, sum_sigma2Po = 0.0;
  double sigma2P, sigma2Po;
  double q1, q2, ssi;
  
  for(int r = 0; r < P.rows(); r++) 
    for(int c = 0; c < P.cols(); c++) 
       {
        double difP = P(r,c)-meanP;
        double difPo = Po(r,c)-meanPo;
        
        sum_sigma2P += difP*difP;
        sum_sigma2Po += difPo*difPo;    
       }
  sigma2P = constante * sum_sigma2P;
  sigma2Po = constante * sum_sigma2Po;  
  q1 = sqrt(sigma2Po) / meanPo;
  q2 = meanP / sqrt(sigma2P);  
  
  ssi = q1*q2;
  
return ssi;
}
//********************************************************
// Speckle Suppression and Mean Preservation Index (SSMPI)
//
// P = datos sin ruido
// Po = estimacion
//********************************************************
double SSMPI(Array<double,2>& P, Array<double,2>& Po)
{
  long int SizeImage = P.rows()*P.cols();
  double constante = 1.0 / (SizeImage - 1.0);
  double meanP = sum(P) * (1.0 / SizeImage );
  double meanPo = sum(Po) * (1.0 / SizeImage );
  double sum_sigma2P = 0.0, sum_sigma2Po = 0.0;
  double sigma2P, sigma2Po;
  double q1, q2, ssmpi;
  
  for(int r = 0; r < P.rows(); r++) 
    for(int c = 0; c < P.cols(); c++) 
       {
        double difP = P(r,c)-meanP;
        double difPo = Po(r,c)-meanPo;
        
        sum_sigma2P += difP*difP;
        sum_sigma2Po += difPo*difPo;    
       }
  sigma2P = constante * sum_sigma2P;
  sigma2Po = constante * sum_sigma2Po;  
  q1 = 1.0 + abs(meanPo - meanP);
  q2 = sqrt(sigma2Po) / sqrt(sigma2P);  

  ssmpi = q1 * q2;
    
return ssmpi;
}
//********************************************************
// Normalized Error Index (NEI)
//
// P = datos sin ruido
// Po = estimacion
//********************************************************
double NEI(Array<double,2>& P, Array<double,2>& Po)
{

double nei = sqrt( sum(pow2(P-Po)) ) / (sqrt( sum(pow2(P)) ) + sqrt( sum(pow2(Po)) ));

return nei;
}

// ************************************************************************
//       funcion principal de la derivada para double
//*************************************************************************
void Derivada( double* & derivIs_h, double* & derivIc_h, double* & Is_h, double* & Ic_h )
{
 // define parametro de regularizacion
  double lambda1 = LAMBDA1, lambda2 = LAMBDA2, lambda3 = LAMBDA3;
  double dxMu11, dyMu12, divMu1, dxMu21, dyMu22, divMu2;
  double dxQ11, dyQ12, divQ1, dxQ21, dyQ22, divQ2, aux;

  // tamano de arreglo
  long int SizeImage = renglones*columnas;

  //gradiente( Is_h, Ic_h, dxIs, dyIs, dxIc, dyIc );
  
  for ( long int r = 1; r < renglones-1; r++ )
    for ( long int c = 1; c < columnas-1; c++ )
      {
       long int idx_r_c = r*columnas + c;
       long int idx_rp1_c = (r + 1)*columnas + c;
       long int idx_rm1_c = (r - 1)*columnas + c;
       long int idx_r_cp1 = r*columnas + c + 1;
       long int idx_r_cm1 = r*columnas + c - 1; 
       long int idx_rm1_cp1 = (r - 1)*columnas + c + 1; 
       long int idx_rp1_cm1 = (r + 1)*columnas + c - 1; 

       // Termino (q- Grad(A))
       Q11[idx_r_c] = Q11[idx_r_c]-dxIc[idx_r_c];//Ic
       Q12[idx_r_c] = Q12[idx_r_c]-dyIc[idx_r_c];
       Q21[idx_r_c] = Q21[idx_r_c]-dxIs[idx_r_c];//Is
       Q22[idx_r_c] = Q22[idx_r_c]-dyIs[idx_r_c];

//           Se obtienen derivadas parciales 
//           para calculo de divergencia
//           long int idx_r_c = r*columnas + c;
//           long int idx_r_cp1 = r*columnas + c + 1;
       dyMu12 = Mu12[idx_r_cp1]-Mu12[idx_r_c];
       dyMu22 = Mu22[idx_r_cp1]-Mu22[idx_r_c];
       dyQ12 = Q12[idx_r_cp1]-Q12[idx_r_c];
       dyQ22 = Q22[idx_r_cp1]-Q22[idx_r_c];           

//           long int idx_r_c = r*columnas + c;
//           long int idx_r_cm1 = r*columnas + c - 1; 
       dyMu12 = Mu12[idx_r_c]-Mu12[idx_r_cm1];
       dyMu22 = Mu22[idx_r_c]-Mu22[idx_r_cm1];
       dyQ12 = Q12[idx_r_c]-Q12[idx_r_cm1];
       dyQ22 = Q22[idx_r_c]-Q22[idx_r_cm1];

//         long int idx_r_c = r*columnas + c;
//     	   long int idx_r_cp1 = r*columnas + c + 1; 
       dyMu12 = Mu12[idx_r_cp1]-Mu12[idx_r_c];
       dyMu22 = Mu22[idx_r_cp1]-Mu22[idx_r_c];
       dyQ12 = Q12[idx_r_cp1]-Q12[idx_r_c];
       dyQ22 = Q22[idx_r_cp1]-Q22[idx_r_c];

//         long int idx_r_c = r*columnas + c;
//	   long int idx_rp1_c = (r + 1)*columnas + c;
       dxMu11 = Mu11[idx_rp1_c]-Mu11[idx_r_c];
       dxMu21 = Mu21[idx_rp1_c]-Mu21[idx_r_c];
       dxQ11 = Q11[idx_rp1_c]-Q11[idx_r_c];
       dxQ21 = Q21[idx_rp1_c]-Q21[idx_r_c];
 
//           long int idx_r_c = r*columnas + c;
//           long int idx_rm1_c = (r - 1)*columnas + c;
        dxMu11 = Mu11[idx_r_c]-Mu11[idx_rm1_c];
        dxMu21 = Mu21[idx_r_c]-Mu21[idx_rm1_c];    
        dxQ11 = Q11[idx_r_c]-Q11[idx_rm1_c];
        dxQ21 = Q21[idx_r_c]-Q21[idx_rm1_c];        

//           long int idx_r_c = r*columnas + c;
//           long int idx_rp1_c = (r + 1)*columnas + c;
        dxMu11 = Mu11[idx_rp1_c]-Mu11[idx_r_c]; 
        dxMu21 = Mu21[idx_rp1_c]-Mu21[idx_r_c];
        dxQ11 = Q11[idx_rp1_c]-Q11[idx_r_c]; 
        dxQ21 = Q21[idx_rp1_c]-Q21[idx_r_c];
       // termina calculo de derivadas parciales
   
       //Obtener div Mu1^k y div Mu2^k       
       divMu1 = -1.0*(dxMu11 + dyMu12);
       divMu2 = -1.0*(dxMu21 + dyMu22);
       
       //Obtener div Q1^k y div Q2^k
       divQ1 = -1.0*(dxQ11 + dyQ12); // Q1 para parte real Ic
       divQ2 = -1.0*(dxQ21 + dyQ22); // Q2 para parte imaginaria Is

      
        // iteracion de descenso de gradiente
        aux = 2.0*lambda3*( Is_h[idx_r_c]*Is_h[idx_r_c] + Ic_h[idx_r_c]*Ic_h[idx_r_c] - 1.0);
        derivIs_h[idx_r_c] = lambda2*(Is_h[idx_r_c] - sn0[idx_r_c]) + aux*Is_h[idx_r_c] + divMu2 + coefR*divQ2;
        derivIc_h[idx_r_c] = lambda1*(Ic_h[idx_r_c] - cs0[idx_r_c]) + aux*Ic_h[idx_r_c] + divMu1 + coefR*divQ1;

}
}
// ************************************************************************
//       Funcion solver para double
//       descenso de Gradiente para ALM
//   Descenso de gradiente para arreglos double
//////////////////////////////////////////////////////////////////////////////
void solve_descenso_gradiente_ALM( double* & Is_h, double* & Ic_h, double* & derivIs_h, double* & derivIc_h, const char* win1, Array<double, 2> dummy, Mat Imagen)
{
  // define parametro de regularizacion
  double lambda1 = LAMBDA1, lambda2 = LAMBDA2, lambda3 = LAMBDA3;
  double dxMu11, dyMu12, divMu1, dxMu21, dyMu22, divMu2;
  double dxQ11, dyQ12, divQ1, dxQ21, dyQ22, divQ2, aux;
  // variables del metodo
  unsigned iter = 0;             // contador de iteraciones   
  // tamano de arreglo
  long int SizeImage = renglones*columnas;
  bool flag = true;

  // Errores para criterio de paro
  double errIc, errIs;

  //condiciones de frontera Neumann para Is, Ic
  boundaryCondALM1( Ic_h, Is_h, renglones, columnas );

  // inicia iteracion del algoritmo
  double Fx0 = Funcional( Is_h, Ic_h, Q11, Q12, Q21, Q22, Mu11, Mu12, Mu21, Mu22 );
  
  // Gradiente de Is, Ic para
  // Termino (q- Grad(A))
  gradiente( Is_h, Ic_h, dxIs, dyIs, dxIc, dyIc );
  
  while ( flag )
    {
      // calcula derivada de la funcional
      Derivada( derivIs_h, derivIc_h, Is_h, Ic_h );


     for(long int idx_r_c = 0; idx_r_c < SizeImage; idx_r_c++) 
       {
        // resguarda para calculo de error
        Is1_h[idx_r_c] = Is_h[idx_r_c];
        Ic1_h[idx_r_c] = Ic_h[idx_r_c];      
       }
  
  for ( long int r = 1; r < renglones-1; r++ )
    for ( long int c = 1; c < columnas-1; c++ )
      {
       long int idx_r_c = r*columnas + c;
       long int idx_rp1_c = (r + 1)*columnas + c;
       long int idx_rm1_c = (r - 1)*columnas + c;
       long int idx_r_cp1 = r*columnas + c + 1;
       long int idx_r_cm1 = r*columnas + c - 1; 
       long int idx_rm1_cp1 = (r - 1)*columnas + c + 1; 
       long int idx_rp1_cm1 = (r + 1)*columnas + c - 1; 
       
       // Termino (q- Grad(A))
       Q11[idx_r_c] = Q11[idx_r_c]-dxIc[idx_r_c];//Ic
       Q12[idx_r_c] = Q12[idx_r_c]-dyIc[idx_r_c];
       Q21[idx_r_c] = Q21[idx_r_c]-dxIs[idx_r_c];//Is
       Q22[idx_r_c] = Q22[idx_r_c]-dyIs[idx_r_c];

//           Se obtienen derivadas parciales 
//           para calculo de divergencia
//           long int idx_r_c = r*columnas + c;
//           long int idx_r_cp1 = r*columnas + c + 1;
       dyMu12 = Mu12[idx_r_cp1]-Mu12[idx_r_c];
       dyMu22 = Mu22[idx_r_cp1]-Mu22[idx_r_c];
       dyQ12 = Q12[idx_r_cp1]-Q12[idx_r_c];
       dyQ22 = Q22[idx_r_cp1]-Q22[idx_r_c];           

//           long int idx_r_c = r*columnas + c;
//           long int idx_r_cm1 = r*columnas + c - 1; 
       dyMu12 = Mu12[idx_r_c]-Mu12[idx_r_cm1];
       dyMu22 = Mu22[idx_r_c]-Mu22[idx_r_cm1];
       dyQ12 = Q12[idx_r_c]-Q12[idx_r_cm1];
       dyQ22 = Q22[idx_r_c]-Q22[idx_r_cm1];

//         long int idx_r_c = r*columnas + c;
//     	   long int idx_r_cp1 = r*columnas + c + 1; 
       dyMu12 = Mu12[idx_r_cp1]-Mu12[idx_r_c];
       dyMu22 = Mu22[idx_r_cp1]-Mu22[idx_r_c];
       dyQ12 = Q12[idx_r_cp1]-Q12[idx_r_c];
       dyQ22 = Q22[idx_r_cp1]-Q22[idx_r_c];

//         long int idx_r_c = r*columnas + c;
//	   long int idx_rp1_c = (r + 1)*columnas + c;
       dxMu11 = Mu11[idx_rp1_c]-Mu11[idx_r_c];
       dxMu21 = Mu21[idx_rp1_c]-Mu21[idx_r_c];
       dxQ11 = Q11[idx_rp1_c]-Q11[idx_r_c];
       dxQ21 = Q21[idx_rp1_c]-Q21[idx_r_c];
 
//           long int idx_r_c = r*columnas + c;
//           long int idx_rm1_c = (r - 1)*columnas + c;
        dxMu11 = Mu11[idx_r_c]-Mu11[idx_rm1_c];
        dxMu21 = Mu21[idx_r_c]-Mu21[idx_rm1_c];    
        dxQ11 = Q11[idx_r_c]-Q11[idx_rm1_c];
        dxQ21 = Q21[idx_r_c]-Q21[idx_rm1_c];        

//           long int idx_r_c = r*columnas + c;
//           long int idx_rp1_c = (r + 1)*columnas + c;
        dxMu11 = Mu11[idx_rp1_c]-Mu11[idx_r_c]; 
        dxMu21 = Mu21[idx_rp1_c]-Mu21[idx_r_c];
        dxQ11 = Q11[idx_rp1_c]-Q11[idx_r_c]; 
        dxQ21 = Q21[idx_rp1_c]-Q21[idx_r_c];
       // termina calculo de derivadas parciales
   
       //Obtener div Mu1^k y div Mu2^k       
       divMu1 = -1.0*(dxMu11 + dyMu12);
       divMu2 = -1.0*(dxMu21 + dyMu22);
       
       //Obtener div Q1^k y div Q2^k
       divQ1 = -1.0*(dxQ11 + dyQ12); // Q1 para parte real Ic
       divQ2 = -1.0*(dxQ21 + dyQ22); // Q2 para parte imaginaria Is

      
        // iteracion de descenso de gradiente
        aux = 2.0*lambda3*( Is_h[idx_r_c]*Is_h[idx_r_c] + Ic_h[idx_r_c]*Ic_h[idx_r_c] - 1.0);
        derivIs_h[idx_r_c] = lambda2*(Is_h[idx_r_c] - sn0[idx_r_c]) + aux*Is_h[idx_r_c] + divMu2 + coefR*divQ2;
        derivIc_h[idx_r_c] = lambda1*(Ic_h[idx_r_c] - cs0[idx_r_c]) + aux*Ic_h[idx_r_c] + divMu1 + coefR*divQ1;
        
        Is_h[idx_r_c] = Is_h[idx_r_c] - TAO*derivIs_h[idx_r_c];
        Ic_h[idx_r_c] = Ic_h[idx_r_c] - TAO*derivIc_h[idx_r_c]; 
      }
       //actualiza gradientes para calculo de Q y Mu
       gradiente( Is_h, Ic_h, dxIs, dyIs, dxIc, dyIc );

       //soft-thresholding operator
       //Actualizar Q1^k
       //Actualizar Q2^k
       actualizaQ(Q11, Q12, Q21, Q22, Mu11, Mu12, Mu21, Mu22, dxIc, dyIc, dxIs, dyIs);      
       //Actualizar Mu1^k
       //Actualizar Mu2^k     
       actualizaMu(Mu11, Mu12, Mu21, Mu22, Q11, Q12, Q21, Q22, dxIc, dyIc, dxIs, dyIs);

//      P = atan2(Is, Ic);

      double Fx = Funcional( Is_h, Ic_h, Q11, Q12, Q21, Q22, Mu11, Mu12, Mu21, Mu22 );
      double difF = fabs(Fx0-Fx);    

      Fx0 = Fx;
      
      // calcula error de la estimación, despliega avances
      // Ic, Is nueva iteracion
      // Ic1, Is1 iteracion anterior     
      error_relativo(&errIc, &errIs, Ic_h, Ic1_h, Is_h, Is1_h);
      //normas_derivadas(&errIc, &errIs, derivIc_h, derivIs_h);

      if ( (iter % 50) == 0 )
        {
          cout << "iteracion : " << iter << " Fx= " << Fx << " ||Is||= " << errIs << " ||Ic||= " << errIc << endl;

     for(int r = 0; r < renglones; r++) 
          for(int c = 0; c < columnas; c++) 
       {
         long int idx_r_c = r*columnas + c;
         dummy(r,c) = (atan2( Is_h[idx_r_c], Ic_h[idx_r_c] ) + M_PI) / (2.0*M_PI);      
       }

          imshow( win1, Imagen );        waitKey( 1 );
        }
        
      // criterios de paro || (difF < epsilon1)
      if ( (iter >= ITER_MAX1) || (errIs < EPSILON1) || (errIc < EPSILON2))
        {
          cout << "iteracion : " << iter << " Fx = " << Fx << " ||Is||= " << errIs << " ||Ic||= " << errIc << endl;
          flag = false;
        }

      // incrementa contador iteracion
      iter++;
    }
    
}
//////////////////////////////////////////////////////////////////////////////
//   Calculo de gradiente para arreglos double
//   Input: Is_h, Ic_h
//   Output: dxIs, dyIs, dxIc, dyIc    derivadas parciales
//////////////////////////////////////////////////////////////////////////////
void gradiente( double* &Is_h, double* &Ic_h, double* &dxIs_h, double* &dyIs_h, double* &dxIc_h, double* &dyIc_h )
{

  // evalua derivadas parciales para funcional
  for ( int r = 0; r < renglones; r++ )
    for ( int c = 0; c < columnas; c++ )
      {
        // campo de gradiente de la informacion
        if ( c == 0 )
          {  
           long int idx_r_c = r*columnas + c;
           long int idx_r_cp1 = r*columnas + c + 1;
           dyIs_h[idx_r_c] = Is_h[idx_r_cp1]-Is_h[idx_r_c];
           dyIc_h[idx_r_c] = Ic_h[idx_r_cp1]-Ic_h[idx_r_c];
           }
        else if ( c == columnas-1 )
          {  
           long int idx_r_c = r*columnas + c;
           long int idx_r_cm1 = r*columnas + c - 1; 
           dyIs_h[idx_r_c] = Is_h[idx_r_c]-Is_h[idx_r_cm1];
           dyIc_h[idx_r_c] = Ic_h[idx_r_c]-Ic_h[idx_r_cm1];
           }
        else
          {  
           long int idx_r_c = r*columnas + c;
     	   long int idx_r_cp1 = r*columnas + c + 1; 
           dyIs_h[idx_r_c] = Is_h[idx_r_cp1]-Is_h[idx_r_c];
           dyIc_h[idx_r_c] = Ic_h[idx_r_cp1]-Ic_h[idx_r_c];
          }
          
        // campo de gradiente de la informacion
        if ( r == 0 )
          {  
           long int idx_r_c = r*columnas + c;
	   long int idx_rp1_c = (r + 1)*columnas + c;
           dxIs_h[idx_r_c] = Is_h[idx_rp1_c]-Is_h[idx_r_c];
           dxIc_h[idx_r_c] = Ic_h[idx_rp1_c]-Ic_h[idx_r_c];
          }
        else if ( r == renglones-1 )
          {  
           long int idx_r_c = r*columnas + c;
           long int idx_rm1_c = (r - 1)*columnas + c;
           dxIs_h[idx_r_c] = Is_h[idx_r_c]-Is_h[idx_rm1_c];
           dxIc_h[idx_r_c] = Ic_h[idx_r_c]-Ic_h[idx_rm1_c];
          }
        else
          {  
           long int idx_r_c = r*columnas + c;
           long int idx_rp1_c = (r + 1)*columnas + c;
           dxIs_h[idx_r_c] = Is_h[idx_rp1_c]-Is_h[idx_r_c]; 
           dxIc_h[idx_r_c] = Ic_h[idx_rp1_c]-Ic_h[idx_r_c];
           }
           
       // termina calculo de derivadas parciales
     }
}
//
// ************************************************************************
//       funcion principal de gradiente2 para double
//*************************************************************************
void gradiente2( double* &Is_h, double* &Ic_h, double* &dxIs_h, double* &dyIs_h, double* &dxIc_h, double* &dyIc_h )
{
  // define variables a a utilizar
  double Ux, Uy, divIs, divIc;
  double V31xIs, V32xIs, V31yIs, V32yIs;
  double V31xIc, V32xIc, V31yIc, V32yIc;

  // evalua primera derivada de fase y
  // terminos de divergencia
  for ( long int r = 0; r < renglones; r++ )
    for ( long int c = 0; c < columnas; c++ )
      {
         long int idx_r_c = r*columnas + c;
	 long int idx_rp1_c = (r + 1)*columnas + c;
	 long int idx_rm1_c = (r - 1)*columnas + c;
	 long int idx_r_cp1 = r*columnas + c + 1;
	 long int idx_r_cm1 = r*columnas + c - 1; 
	 long int idx_rm1_cp1 = (r - 1)*columnas + c + 1; 
	 long int idx_rp1_cm1 = (r + 1)*columnas + c - 1;
         long int idx_rp1_cp1 = (r + 1)*columnas + c + 1;//se añade
         long int idx_rm1_cm1 = (r - 1)*columnas + c - 1;//se añade 
       // termino para divergencia de nabla phi / |nabla phi|
        // procesa condiciones de frontera, eje x
        if ( r == renglones-1 || c == 0 || c == columnas-1 ) {  V31xIs = 0.0; V31xIc = 0.0; }
        else
          {
            Ux = Is_h[idx_rp1_c]-Is_h[idx_r_c];
            Uy = minMod( 0.5*(Is_h[idx_rp1_cp1] - Is_h[idx_rp1_cm1]), 0.5*(Is_h[idx_r_cp1] - Is_h[idx_r_cm1]));
            V31xIs = Ux;
            Ux = Ic_h[idx_rp1_c]-Ic_h[idx_r_c];
            Uy = minMod( 0.5*(Ic_h[idx_rp1_cp1] - Ic_h[idx_rp1_cm1]), 0.5*(Ic_h[idx_r_cp1] - Ic_h[idx_r_cm1]));
            V31xIc = Ux;
          }  

        if ( r == 0 || c == 0 || c == columnas-1 ) {  V32xIs = 0.0; V32xIc = 0.0; }
        else
          {           
            Ux = Is_h[idx_r_c]-Is_h[idx_rm1_c];
            Uy = minMod( 0.5*(Is_h[idx_r_cp1] - Is_h[idx_r_cm1]), 0.5*(Is_h[idx_rm1_cp1] - Is_h[idx_rm1_cm1]) );
            V32xIs = Ux;// / sqrt( Ux*Ux + Uy*Uy + beta );
            Ux = Ic_h[idx_r_c]-Ic_h[idx_rm1_c];
            Uy = minMod( 0.5*(Ic_h[idx_r_cp1] - Ic_h[idx_r_cm1]), 0.5*(Ic_h[idx_rm1_cp1] - Ic_h[idx_rm1_cm1]) );
            V32xIc = Ux;// / sqrt( Ux*Ux + Uy*Uy + beta );
          }  
      
        // procesa condiciones de frontera, eje y
        if ( c == columnas-1 || r == 0 || r == renglones-1) { V31yIs = 0.0; V31yIc = 0.0; }
        else
          {            
            Ux = minMod( 0.5*(Is_h[idx_rp1_cp1] - Is_h[idx_rm1_cp1]), 0.5*(Is_h[idx_rp1_c] - Is_h[idx_rm1_c]) );
            Uy = Is_h[idx_r_cp1] - Is_h[idx_r_c];
            V31yIs = Uy;// / sqrt( Ux*Ux + Uy*Uy + beta );
            Ux = minMod( 0.5*(Ic_h[idx_rp1_cp1] - Ic_h[idx_rm1_cp1]), 0.5*(Ic_h[idx_rp1_c] - Ic_h[idx_rm1_c]) );
            Uy = Ic_h[idx_r_cp1] - Ic_h[idx_r_c];
            V31yIc = Uy;// / sqrt( Ux*Ux + Uy*Uy + beta );
          }  

        if ( c == 0 || r == 0 || r == renglones-1) {  V32yIs = 0.0; V32yIc = 0.0; }
        else
          {
            Ux = minMod( 0.5*(Is_h[idx_rp1_c] - Is_h[idx_rm1_c]), 0.5*(Is_h[idx_rp1_cm1] - Is_h[idx_rm1_cm1]) );
            Uy = Is_h[idx_r_c] - Is_h[idx_r_cm1];
            V32yIs = Uy;// / sqrt( Ux*Ux + Uy*Uy + beta );
            Ux = minMod( 0.5*(Ic_h[idx_rp1_c] - Ic_h[idx_rm1_c]), 0.5*(Ic_h[idx_rp1_cm1] - Ic_h[idx_rm1_cm1]) );
            Uy = Ic_h[idx_r_c] - Ic_h[idx_r_cm1];
            V32yIc = Uy;// / sqrt( Ux*Ux + Uy*Uy + beta );
          }

       //divIs = (V31xIs-V32xIs) + (V31yIs-V32yIs);
       //divIc = (V31xIc-V32xIc) + (V31yIc-V32yIc);
       
       dxIs_h[idx_r_c] = V31xIs-V32xIs;
       dyIs_h[idx_r_c] = V31yIs-V32yIs;
       dxIc_h[idx_r_c] = V31xIc-V32xIc;
       dyIc_h[idx_r_c] = V31yIc-V32yIc;

      } 
}

//   
//
//////////////////////////////////////////////////////////////////////////////
//            soft-thresholding operator
//   Input: Mu11, Mu12, dxIs, dyIs, dxIc, dyIc    derivadas parciales
//   Output: Q11, Q12, 
//////////////////////////////////////////////////////////////////////////////
void actualizaQ( double* &Q11, double* &Q12, double* &Q21, double* &Q22, double* &Mu11, double* &Mu12, double* &Mu21, double* &Mu22, double* &dxIc, double* &dyIc, double* &dxIs, double* &dyIs)
{
// calculo de normas
 double sum1 = 0.0, sum2 = 0.0, w11, w12, w21, w22;
   for ( long int r = 0; r < renglones; r++ )
    for ( long int c = 0; c < columnas; c++ )
      {
        long int idx_r_c = r*columnas + c;
        w11 = coefR*dxIc[idx_r_c]-Mu11[idx_r_c];
        w12 = coefR*dyIc[idx_r_c]-Mu12[idx_r_c];
        sum1 += w11*w11 + w12*w12;
        w21 = coefR*dxIs[idx_r_c]-Mu21[idx_r_c];
        w22 = coefR*dyIs[idx_r_c]-Mu22[idx_r_c];        
        sum2 += w21*w21 + w22*w22;
      }
  
   double norma1 = sqrt(sum1);
   double norma2 = sqrt(sum2);
   
   //soft-thresholding operator
   for ( long int r = 0; r < renglones; r++ )
    for ( long int c = 0; c < columnas; c++ )
      {
       long int idx_r_c = r*columnas + c;     
    if( norma1 > 1.0)
      {
        Q11[idx_r_c] = (1.0/coefR)*(1.0 - (1.0/norma1))*(coefR*dxIc[idx_r_c]-Mu11[idx_r_c]);//w11[idx_r_c];
        Q12[idx_r_c] = (1.0/coefR)*(1.0 - (1.0/norma1))*(coefR*dyIc[idx_r_c]-Mu12[idx_r_c]);//w12[idx_r_c];
      }
       else
         {
          Q11[idx_r_c] = 0.0;
          Q12[idx_r_c] = 0.0;
         }

    if( norma2 > 1.0)
      {
        Q21[idx_r_c] = (1.0/coefR)*(1.0 - (1.0/norma2))*(coefR*dxIs[idx_r_c]-Mu21[idx_r_c]);//w21[idx_r_c];
        Q22[idx_r_c] = (1.0/coefR)*(1.0 - (1.0/norma2))*(coefR*dyIs[idx_r_c]-Mu22[idx_r_c]);//w22[idx_r_c];
      }
       else
         {
          Q21[idx_r_c] = 0.0;
          Q22[idx_r_c] = 0.0;
         }   
      }     
}
//////////////////////////////////////////////////////////////////////////////
//                    parametro Mu
//   Input: Mu11, Mu12, Q11, Q12, dxIs, dyIs, dxIc, dyIc    derivadas parciales
//   Output: Mu11, Mu12
//////////////////////////////////////////////////////////////////////////////
void actualizaMu(double* &Mu11, double* &Mu12, double* &Mu21, double* &Mu22, double* &Q11, double* &Q12, double* &Q21, double* &Q22, double* &dxIc, double* &dyIc, double* &dxIs, double* &dyIs)
{
   for ( long int r = 0; r < renglones; r++ )
    for ( long int c = 0; c < columnas; c++ )
      {
       long int idx_r_c = r*columnas + c;
       Mu11[idx_r_c] = Mu11[idx_r_c] + coefR*(Q11[idx_r_c] - dxIc[idx_r_c]);
       Mu12[idx_r_c] = Mu12[idx_r_c] + coefR*(Q12[idx_r_c] - dyIc[idx_r_c]);
       Mu21[idx_r_c] = Mu21[idx_r_c] + coefR*(Q21[idx_r_c] - dxIs[idx_r_c]);
       Mu22[idx_r_c] = Mu22[idx_r_c] + coefR*(Q22[idx_r_c] - dyIs[idx_r_c]);
      }
}


void solve_gradiente_nesterov_ALM( double* &Is_h, double* &Ic_h, double* &derivIs_h, double* &derivIc_h, const char* win1, Array<double, 2> dummy, Mat Imagen )
{
 // define parametro de regularizacion
  double lambda1 = LAMBDA1, lambda2 = LAMBDA2, lambda3 = LAMBDA3;
  double dxMu11, dyMu12, divMu1, dxMu21, dyMu22, divMu2;
  double dxQ11, dyQ12, divQ1, dxQ21, dyQ22, divQ2, aux;


  // variables del metodo
  double errIs, errIc, errdIs, errdIc;
  unsigned iter = 0;             // contador de iteraciones   
  bool flag = true;
  long int SizeImage = renglones*columnas;


  double theta0 = 1.0, theta1;        // valores iniciales
  
  //condiciones de frontera Neumann para Is, Ic
  boundaryCondALM1( Ic_h, Is_h, renglones, columnas );
  
  // resguarda Is2 = Is
     for(long int idx_r_c = 0; idx_r_c < SizeImage; idx_r_c++) 
       {
        // resguarda Is2 = Is, Ic2 = Ic
        Is2_h[idx_r_c] = Is_h[idx_r_c];
        Ic2_h[idx_r_c] = Ic_h[idx_r_c];
       }  

  double Fx0 = Funcional( Is_h, Ic_h, Q11, Q12, Q21, Q22, Mu11, Mu12, Mu21, Mu22 );

  // Gradiente de Is, Ic para
  // Termino (q- Grad(A))
  gradiente( Is_h, Ic_h, dxIs, dyIs, dxIc, dyIc );

  while ( flag )
    {  
    theta1 = 0.5*( 1.0 + sqrt(1.0 + 4.0*theta0*theta0) );
    double gamma = (theta0-1.0)/theta1;
    
    //Derivada de iteracion actual
    Derivada(derivIs_h, derivIc_h, Is_h, Ic_h);
    
     // resguarda Is0, Ic0 para calculo de error
     // Is11, Ic11 
     //for(long int idx_r_c = 0; idx_r_c < SizeImage; idx_r_c++) 
     for(long int r = 1; r < renglones-1; r++) 
      for(long int c = 1; c < columnas-1; c++) 
       {
        long int idx_r_c = r*columnas + c; 
//        Is11_h[idx_r_c] = Is_h[idx_r_c] + gamma*(Is_h[idx_r_c] - Is0_h[idx_r_c]);
//        Ic11_h[idx_r_c] = Ic_h[idx_r_c] + gamma*(Ic_h[idx_r_c] - Ic0_h[idx_r_c]);  

        
        // resguarda para calculo de error
        Is0_h[idx_r_c] = Is_h[idx_r_c];
        Ic0_h[idx_r_c] = Ic_h[idx_r_c];  
        
        // Is1 = Is - TAO*derivIs
        Is11_h[idx_r_c] = Is_h[idx_r_c]-TAO*derivIs_h[idx_r_c];
        Ic11_h[idx_r_c] = Ic_h[idx_r_c]-TAO*derivIc_h[idx_r_c];  
                
////        // Is, Ic auxiliar = extrapolacion
//        Is11_h[idx_r_c] = Is11_h[idx_r_c] + gamma*(Is11_h[idx_r_c] - Is0_h[idx_r_c]);
//        Ic11_h[idx_r_c] = Ic11_h[idx_r_c] + gamma*(Ic11_h[idx_r_c] - Ic0_h[idx_r_c]); 

        // codigo para gradiente evaluado en Is11, Ic11
       //long int idx_r_c = r*columnas + c;
       long int idx_rp1_c = (r + 1)*columnas + c;
       long int idx_rm1_c = (r - 1)*columnas + c;
       long int idx_r_cp1 = r*columnas + c + 1;
       long int idx_r_cm1 = r*columnas + c - 1; 
       long int idx_rm1_cp1 = (r - 1)*columnas + c + 1; 
       long int idx_rp1_cm1 = (r + 1)*columnas + c - 1; 

       // Grad(A)
        // campo de gradiente de la informacion
        if ( c == 0 )
          {  
//           long int idx_r_c = r*columnas + c;
//           long int idx_r_cp1 = r*columnas + c + 1;
           dyIs[idx_r_c] = Is11_h[idx_r_cp1]-Is11_h[idx_r_c];
           dyIc[idx_r_c] = Ic11_h[idx_r_cp1]-Ic11_h[idx_r_c];
           }
        else if ( c == columnas-1 )
          {  
//           long int idx_r_c = r*columnas + c;
//           long int idx_r_cm1 = r*columnas + c - 1; 
           dyIs[idx_r_c] = Is11_h[idx_r_c]-Is11_h[idx_r_cm1];
           dyIc[idx_r_c] = Ic11_h[idx_r_c]-Ic11_h[idx_r_cm1];
           }
        else
          {  
//           long int idx_r_c = r*columnas + c;
//     	   long int idx_r_cp1 = r*columnas + c + 1; 
           dyIs[idx_r_c] = Is11_h[idx_r_cp1]-Is11_h[idx_r_c];
           dyIc[idx_r_c] = Ic11_h[idx_r_cp1]-Ic11_h[idx_r_c];
          }
          
        // campo de gradiente de la informacion
        if ( r == 0 )
          {  
//           long int idx_r_c = r*columnas + c;
//	   long int idx_rp1_c = (r + 1)*columnas + c;
           dxIs[idx_r_c] = Is11_h[idx_rp1_c]-Is11_h[idx_r_c];
           dxIc[idx_r_c] = Ic11_h[idx_rp1_c]-Ic11_h[idx_r_c];
          }
        else if ( r == renglones-1 )
          {  
//           long int idx_r_c = r*columnas + c;
//           long int idx_rm1_c = (r - 1)*columnas + c;
           dxIs[idx_r_c] = Is11_h[idx_r_c]-Is11_h[idx_rm1_c];
           dxIc[idx_r_c] = Ic11_h[idx_r_c]-Ic11_h[idx_rm1_c];
          }
        else
          {  
//           long int idx_r_c = r*columnas + c;
//           long int idx_rp1_c = (r + 1)*columnas + c;
           dxIs[idx_r_c] = Is11_h[idx_rp1_c]-Is11_h[idx_r_c]; 
           dxIc[idx_r_c] = Ic11_h[idx_rp1_c]-Ic11_h[idx_r_c];
           }

       // Termino (q- Grad(A))
       Q11[idx_r_c] = Q11[idx_r_c]-dxIc[idx_r_c];//Ic
       Q12[idx_r_c] = Q12[idx_r_c]-dyIc[idx_r_c];
       Q21[idx_r_c] = Q21[idx_r_c]-dxIs[idx_r_c];//Is
       Q22[idx_r_c] = Q22[idx_r_c]-dyIs[idx_r_c];

//           Se obtienen derivadas parciales 
//           para calculo de divergencia
//           long int idx_r_c = r*columnas + c;
//           long int idx_r_cp1 = r*columnas + c + 1;
       dyMu12 = Mu12[idx_r_cp1]-Mu12[idx_r_c];
       dyMu22 = Mu22[idx_r_cp1]-Mu22[idx_r_c];
       dyQ12 = Q12[idx_r_cp1]-Q12[idx_r_c];
       dyQ22 = Q22[idx_r_cp1]-Q22[idx_r_c];           

//           long int idx_r_c = r*columnas + c;
//           long int idx_r_cm1 = r*columnas + c - 1; 
       dyMu12 = Mu12[idx_r_c]-Mu12[idx_r_cm1];
       dyMu22 = Mu22[idx_r_c]-Mu22[idx_r_cm1];
       dyQ12 = Q12[idx_r_c]-Q12[idx_r_cm1];
       dyQ22 = Q22[idx_r_c]-Q22[idx_r_cm1];

//         long int idx_r_c = r*columnas + c;
//     	   long int idx_r_cp1 = r*columnas + c + 1; 
       dyMu12 = Mu12[idx_r_cp1]-Mu12[idx_r_c];
       dyMu22 = Mu22[idx_r_cp1]-Mu22[idx_r_c];
       dyQ12 = Q12[idx_r_cp1]-Q12[idx_r_c];
       dyQ22 = Q22[idx_r_cp1]-Q22[idx_r_c];

//         long int idx_r_c = r*columnas + c;
//	   long int idx_rp1_c = (r + 1)*columnas + c;
       dxMu11 = Mu11[idx_rp1_c]-Mu11[idx_r_c];
       dxMu21 = Mu21[idx_rp1_c]-Mu21[idx_r_c];
       dxQ11 = Q11[idx_rp1_c]-Q11[idx_r_c];
       dxQ21 = Q21[idx_rp1_c]-Q21[idx_r_c];
 
//           long int idx_r_c = r*columnas + c;
//           long int idx_rm1_c = (r - 1)*columnas + c;
        dxMu11 = Mu11[idx_r_c]-Mu11[idx_rm1_c];
        dxMu21 = Mu21[idx_r_c]-Mu21[idx_rm1_c];    
        dxQ11 = Q11[idx_r_c]-Q11[idx_rm1_c];
        dxQ21 = Q21[idx_r_c]-Q21[idx_rm1_c];        

//           long int idx_r_c = r*columnas + c;
//           long int idx_rp1_c = (r + 1)*columnas + c;
        dxMu11 = Mu11[idx_rp1_c]-Mu11[idx_r_c]; 
        dxMu21 = Mu21[idx_rp1_c]-Mu21[idx_r_c];
        dxQ11 = Q11[idx_rp1_c]-Q11[idx_r_c]; 
        dxQ21 = Q21[idx_rp1_c]-Q21[idx_r_c];
       // termina calculo de derivadas parciales
   
       //Obtener div Mu1^k y div Mu2^k       
       divMu1 = -1.0*(dxMu11 + dyMu12);
       divMu2 = -1.0*(dxMu21 + dyMu22);
       
       //Obtener div Q1^k y div Q2^k
       divQ1 = -1.0*(dxQ11 + dyQ12); // Q1 para parte real Ic
       divQ2 = -1.0*(dxQ21 + dyQ22); // Q2 para parte imaginaria Is

      
        //Gradiente en la posicion anticipada
        aux = 2.0*lambda3*( Is11_h[idx_r_c]*Is11_h[idx_r_c] + Ic11_h[idx_r_c]*Ic11_h[idx_r_c] - 1.0);
        derivIs_h[idx_r_c] = lambda2*(Is11_h[idx_r_c] - sn0[idx_r_c]) + aux*Is11_h[idx_r_c] + divMu2 + coefR*divQ2;
        derivIc_h[idx_r_c] = lambda1*(Ic11_h[idx_r_c] - cs0[idx_r_c]) + aux*Ic11_h[idx_r_c] + divMu1 + coefR*divQ1;
        // termina codigo de gradiente

//        //actualizacion de variables 
        Is_h[idx_r_c] = Is11_h[idx_r_c] - TAO*derivIs_h[idx_r_c];
        Ic_h[idx_r_c] = Ic11_h[idx_r_c] - TAO*derivIc_h[idx_r_c];    
        }
       
      double Fx = Funcional( Is_h, Ic_h, Q11, Q12, Q21, Q22, Mu11, Mu12, Mu21, Mu22 );
      double difF = fabs(Fx0-Fx);  
      
//      gradiente( Is_h, Ic_h, dxIs, dyIs, dxIc, dyIc );
//      Derivada(derivIs_h, derivIc_h, Is_h, Ic_h);
//      // reset a las condiciones del descenso, update variables
//      // Found Comput Math (2015) 15:715–732
//      //double suma = sum( derivP*(P1-P2) );
//      double suma_Is = 0.0, suma_Ic = 0.0;
//      for(long int idx_r_c = 0; idx_r_c < SizeImage; idx_r_c++) 
//        {
//         suma_Is += derivIs_h[idx_r_c]*(Is11_h[idx_r_c]-Is2_h[idx_r_c]);
//         suma_Ic += derivIc_h[idx_r_c]*(Ic11_h[idx_r_c]-Ic2_h[idx_r_c]);
//        }      
//      if ( (suma_Is > 0.0) || (suma_Ic > 0.0) || (Fx > Fx0) ) 
//        {//    P2 = P;      theta0 = 1.0;    
//         for(long int idx_r_c = 0; idx_r_c < SizeImage; idx_r_c++) 
//           {
//            Is2_h[idx_r_c] = Is_h[idx_r_c];
//            Ic2_h[idx_r_c] = Ic_h[idx_r_c];
//           }
//         theta0 = 1.0; 
//        }
//      else
//        {//    P2 = P1;      theta0 = theta1;    
//         for(long int idx_r_c = 0; idx_r_c < SizeImage; idx_r_c++) 
//           {
//            Is2_h[idx_r_c] = Is11_h[idx_r_c];
//            Ic2_h[idx_r_c] = Ic11_h[idx_r_c];
//           }
//        theta0 = theta1; 
//        }  
      theta0 = theta1; 
   
      Fx0 = Fx;

       //actualiza gradientes para calculo de Q y Mu
       gradiente( Is_h, Ic_h, dxIs, dyIs, dxIc, dyIc );

       //soft-thresholding operator
       //Actualizar Q1^k
       //Actualizar Q2^k
       actualizaQ(Q11, Q12, Q21, Q22, Mu11, Mu12, Mu21, Mu22, dxIc, dyIc, dxIs, dyIs);      
       //Actualizar Mu1^k
       //Actualizar Mu2^k     
       actualizaMu(Mu11, Mu12, Mu21, Mu22, Q11, Q12, Q21, Q22, dxIc, dyIc, dxIs, dyIs);
      
      // calcula error de la estimación, despliega avances
      error_relativo(&errIc, &errIs, Ic_h, Ic0_h, Is_h, Is0_h);
      normas_derivadas(&errdIc, &errdIs, derivIc_h, derivIs_h);
      if ( (iter % 50) == 0 )
        {
          cout << "iteracion : " << iter << " Fx= " << Fx << " ||Is||= " << errIs << " ||Ic||= " << errIc << endl;

        for(int r = 0; r < renglones; r++) 
          for(int c = 0; c < columnas; c++) 
            {
             long int idx_r_c = r*columnas + c;
             dummy(r,c) = (atan2( Is_h[idx_r_c], Ic_h[idx_r_c] ) + M_PI) / (2.0*M_PI);      
            }
          imshow( win1, Imagen );        waitKey( 1 );
        }
        
      // criterios de paro || (difF < epsilon1)
      if ( (iter >= ITER_MAX1)  || (errIs < EPSILON1) || (errIc < EPSILON2) || (errdIs < EPSILON1) || (errdIc < EPSILON2) || ( difF < EPSILON1*errIs) || ( difF < EPSILON1*errIc))
        {
          cout << "iteracion : " << iter << " Fx = " << Fx << " ||Is||= " << errIs << " ||Ic||= " << errIc << endl;
          flag = false;
        }

      // incrementa contador iteracion
      iter++;
    }

}

void solve_punto_fijo_ALM(double* &Is_h, double* &Ic_h, double* &Is1_h, double* &Ic1_h, const char* win1, Array<double, 2> dummy, Mat Imagen)
{

  // variables del metodo
  double errIs, errIc;
  unsigned iter = 0;             // contador de iteraciones   
  bool flag = true;
  long int SizeImage = renglones*columnas;
  

  double Fx0 = Funcional( Is_h, Ic_h, Q11, Q12, Q21, Q22, Mu11, Mu12, Mu21, Mu22 );

  while ( flag )
    {
     // resguarda Is0, Ic0 para calculo de error
     for(long int idx_r_c = 0; idx_r_c < SizeImage; idx_r_c++) 
       {
        // resguarda para calculo de error
        Is0_h[idx_r_c] = Is_h[idx_r_c];
        Ic0_h[idx_r_c] = Ic_h[idx_r_c];             
       }
      

      // calcula iteracion de Gauss-Seidel
      // retorna solucion actualizada en Is, Ic
      punto_Fijo_TV_ALM(Is_h, Ic_h, Is1_h, Ic1_h);
      //solve_descenso_gradiente_ALM(Is_h, Ic_h, Is1_h, Ic1_h);

      //P = atan2(Is, Ic);

      double Fx = Funcional( Is_h, Ic_h, Q11, Q12, Q21, Q22, Mu11, Mu12, Mu21, Mu22 );
      double difF = fabs(Fx0-Fx);  

      Fx0 = Fx;

       //actualiza gradientes para calculo de Q y Mu
       gradiente( Is_h, Ic_h, dxIs, dyIs, dxIc, dyIc );

       //soft-thresholding operator
       //Actualizar Q1^k
       //Actualizar Q2^k
       actualizaQ(Q11, Q12, Q21, Q22, Mu11, Mu12, Mu21, Mu22, dxIc, dyIc, dxIs, dyIs);      
       //Actualizar Mu1^k
       //Actualizar Mu2^k     
       actualizaMu(Mu11, Mu12, Mu21, Mu22, Q11, Q12, Q21, Q22, dxIc, dyIc, dxIs, dyIs);
      
      // calcula error de la estimación, despliega avances
      error_relativo(&errIc, &errIs, Ic_h, Ic0_h, Is_h, Is0_h);
      //normas_derivadas(&errIc, &errIs, derivIc_h, derivIs_h);
      if ( (iter % 50) == 0 )
        {
          cout << "iteracion : " << iter << " Fx= " << Fx << " ||Is||= " << errIs << " ||Ic||= " << errIc << endl;

        for(int r = 0; r < renglones; r++) 
          for(int c = 0; c < columnas; c++) 
            {
             long int idx_r_c = r*columnas + c;
             dummy(r,c) = (atan2( Is_h[idx_r_c], Ic_h[idx_r_c] ) + M_PI) / (2.0*M_PI);      
            }
          imshow( win1, Imagen );        waitKey( 1 );
        }
        
      // criterios de paro || (difF < epsilon1)
      if ( (iter >= ITER_MAX1)  || (errIs < EPSILON1) || (errIc < EPSILON2))
        {
          cout << "iteracion : " << iter << " Fx = " << Fx << " ||Is||= " << errIs << " ||Ic||= " << errIc << endl;
          flag = false;
        }

      // incrementa contador iteracion
      iter++;
    }
}
