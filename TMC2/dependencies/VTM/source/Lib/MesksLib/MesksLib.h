#ifndef __MesksLibH__
#define __MesksLibH__
#include <Python.h>
//#include <torch/script.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <condition_variable>
#include "../CommonLib/CodingStructure.h"
#include "../CommonLib/Picture.h"
using namespace std;

extern unsigned int*** occupancyData;
extern int			   picWidth;
extern int			   picHeight;
extern int             mapWidth;
extern int             mapHeight;
extern int             mapFrameCount;

void initOccupancyData( int width, int height );
void deleteOccupancyData();
void fillOccupancyData( int frameNum, vector<uint32_t> occupancyMap );


//** VVC variable **//
extern int realtimeFrameCount;
extern int OorGorA;

double nDecimalDouble(const double &dbNum, int n);


//** fast partition variable **//
extern bool* sideBlock_key;
extern double ***      v_side;
extern double ***      h_side;
extern string CUkeys;
extern PyObject *     pModule, *pFunc_nnDriver;
extern double         sum_time;

void deleteSideBlockKey();
void initSideBlockKey();
void deleteSideArray();
void initSideArray();
void MCCNNInit();
void MCCNNDestroy();
//int  nnPredict(double **&TOriFeatures, double **&ResiFeatures, double **&OccuFeatures, double inputQP, int modelType);
int  nnPredict(const CodingStructure *bestCS, Partitioner *partitioner, int modelType);
bool isSkipPartition( int width, int height, int X, int Y, int currMode );
void checkSide();
void checkSide(int X, int Y);


#endif  // __MesksLibH__