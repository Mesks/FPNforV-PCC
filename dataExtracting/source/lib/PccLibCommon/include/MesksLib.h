#ifndef __MesksLibH__
#define __MesksLibH__
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
using namespace std;

extern unsigned int*** occupancyData;
extern int             mapWidth;
extern int             mapHeight;
extern int             picWidth;
extern int             picHeight;
extern int             mapFrameCount;

void initOccupancyData( int width, int height );
void deleteOccupancyData();
void fillOccupancyData( int frameNum, vector<uint32_t> occupancyMap );


//** VVC variable **//
//extern int realtimeFrameCount;
extern int OorGorA;

double nDecimalDouble( const double& dbNum, int n);


//** Features Extracting **//
extern int            stemp;
extern string CUkeys;
extern vector<double> QPList;
extern ofstream GeomI_TOri_file;
extern ofstream GeomI_Resi_file;
extern ofstream GeomI_Occu_file;
extern ofstream GeomI_y_QP_file;
extern ofstream GeomP_TOri_file;
extern ofstream GeomP_Resi_file;
extern ofstream GeomP_Occu_file;
extern ofstream GeomP_y_QP_file;
extern ofstream AttrI_TOri_file;
extern ofstream AttrI_Resi_file;
extern ofstream AttrI_Occu_file;
extern ofstream AttrI_y_QP_file;
extern ofstream AttrP_TOri_file;
extern ofstream AttrP_Resi_file;
extern ofstream AttrP_Occu_file;
extern ofstream AttrP_y_QP_file;

extern int* sideBlock_key;

void openFeaturesExtracting();
void closeFeaturesExtracting();
void deleteSideBlockKey();
void initSideBlockKey();

//void occupancyDciInit( int height, int width, int frameNum, string occupancyName );
//void checkData( int frameNum, int height, int width );

#endif  // __MesksLibH__