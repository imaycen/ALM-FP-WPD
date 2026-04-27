# ALM-FP-WPD


Programa que resuelve el modelo de WPD vía la reformulación con el Metodo de Lagrangiano Aumentado (ALM) realiza K iteraciones de Gauss-Seidel.

// It is necessary to create a folder: imagenes, where the program stores the generated output

//*************************************************************************
//
// Program that solves the filtering model using fixed point
// and gradient descent for the Augmented Lagrangian Method (ALM)
// performs K Gauss-Seidel iterations
//
// Author       : Iván de Jesús May-Cen
// Language     : C++
// Compiler     : g++
// Environment  : 
// Revisions
//   Initial    : 2024-03-01 09:36:44 
//   Last       : 
// Se añaden funciones para el calculo de metricas
//
//  para compilar
//    g++ -O2 ALMfiltradoPF.cpp -o test -lrt -lblitz `pkg-config --cflags opencv4` `pkg-config --libs opencv4`
//  para ejecutar
//    ./test imgname.png coefR lambda1 lambda2 lambda3
// 
//*************************************************************************
