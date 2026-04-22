# ALM-FP-WPD


Programa que resuelve el modelo de WPD vía la reformulación con el Metodo de Lagrangiano Aumentado (ALM) realiza K iteraciones de Gauss-Seidel.


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
// Se añaden funciones para el calculo de metricas
//
//  para compilar
//    g++ -O2 ALMfiltradoPF.cpp -o test -lrt -lblitz `pkg-config --cflags opencv4` `pkg-config --libs opencv4`
//  para ejecutar
//    ./test imgname.png coefR lambda1 lambda2 lambda3
// 
//*************************************************************************
