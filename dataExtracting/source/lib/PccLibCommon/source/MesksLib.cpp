#ifndef __MesksLibCPP__
#define __MesksLibCPP__

#include "MesksLib.h"

unsigned int*** occupancyData = NULL;
int             mapWidth      = 0;
int             mapHeight     = 0;
int             picWidth      = 0;
int             picHeight     = 0;
int             mapFrameCount = 0;
//int             realtimeFrameCount = -1;
int             OorGorA            = -2;
int             stemp               = 0;
int*           sideBlock_key      = NULL;

vector<double> QPList;
string CUkeys;
ofstream GeomI_TOri_file;
ofstream GeomI_Resi_file;
ofstream GeomI_Occu_file;
ofstream GeomI_y_QP_file;
ofstream GeomP_TOri_file;
ofstream GeomP_Resi_file;
ofstream GeomP_Occu_file;
ofstream GeomP_y_QP_file;
ofstream AttrI_TOri_file;
ofstream AttrI_Resi_file;
ofstream AttrI_Occu_file;
ofstream AttrI_y_QP_file;
ofstream AttrP_TOri_file;
ofstream AttrP_Resi_file;
ofstream AttrP_Occu_file;
ofstream AttrP_y_QP_file;

void initOccupancyData( int width, int height) {
  mapWidth      = width;
  mapHeight     = height;
  occupancyData = new unsigned int**[mapFrameCount];
  for ( int i = 0; i < mapFrameCount; i++ ) {
    occupancyData[i] = new unsigned int*[mapWidth];
    for ( int j = 0; j < mapWidth; j++ ) {
      // cout << "mapFrameCount: " << mapFrameCount << ", mapWidth: " << mapWidth << ", mapHeight: "<<mapHeight<<endl;
      occupancyData[i][j] = new unsigned int[mapHeight];
    }
  }
}

void deleteSideBlockKey() {
  if ( sideBlock_key != NULL ) {
    delete[] sideBlock_key;
    sideBlock_key = NULL;
  }
}

void initSideBlockKey() {
  deleteSideBlockKey();

  int blockNum  = (int)( picWidth / 64 ) * (int)( picHeight / 64 );
  sideBlock_key = new int[blockNum];
  for ( int i = 0; i < blockNum; i++ ) sideBlock_key[i] = -1;
}

void deleteOccupancyData() {
  for ( int i = 0; i < mapFrameCount; i++ ) {
    for ( int j = 0; j < mapWidth; j++ ) {
      delete[] occupancyData[i][j];
      occupancyData[i][j] = NULL;
    }
    delete[] occupancyData[i];
    occupancyData[i] = NULL;
  }
  delete[] occupancyData;
  occupancyData = NULL;
}

void fillOccupancyData( int frameNum, vector<uint32_t> occupancyMap ) {
  for ( int i = 0; i < mapHeight; i++ ) {
    for ( int j = 0; j < mapWidth; j++ ) { 
        //cout << "i * mapWidth + j: " << i * mapWidth + j << ", occupancyMap.size: " << occupancyMap.size()<<endl;
        occupancyData[frameNum][j][i] = occupancyMap[i * mapWidth + j];
    }
  }
}

void openFeaturesExtracting() {
  if ( OorGorA == 0 ) {
    GeomI_TOri_file.open( "../__extraFeatures/GeomI_TOri.csv", ios::app );
    GeomI_Resi_file.open( "../__extraFeatures/GeomI_Resi.csv", ios::app );
    GeomI_Occu_file.open( "../__extraFeatures/GeomI_Occu.csv", ios::app );
    GeomI_y_QP_file.open( "../__extraFeatures/GeomI_y_QP.csv", ios::app );
    GeomP_TOri_file.open( "../__extraFeatures/GeomP_TOri.csv", ios::app );
    GeomP_Resi_file.open( "../__extraFeatures/GeomP_Resi.csv", ios::app );
    GeomP_Occu_file.open( "../__extraFeatures/GeomP_Occu.csv", ios::app );
    GeomP_y_QP_file.open( "../__extraFeatures/GeomP_y_QP.csv", ios::app );
  } else if ( OorGorA > 0 ) {
    AttrI_TOri_file.open( "../__extraFeatures/AttrI_TOri.csv", ios::app );
    AttrI_Resi_file.open( "../__extraFeatures/AttrI_Resi.csv", ios::app );
    AttrI_Occu_file.open( "../__extraFeatures/AttrI_Occu.csv", ios::app );
    AttrI_y_QP_file.open( "../__extraFeatures/AttrI_y_QP.csv", ios::app );
    AttrP_TOri_file.open( "../__extraFeatures/AttrP_TOri.csv", ios::app );
    AttrP_Resi_file.open( "../__extraFeatures/AttrP_Resi.csv", ios::app );
    AttrP_Occu_file.open( "../__extraFeatures/AttrP_Occu.csv", ios::app );
    AttrP_y_QP_file.open( "../__extraFeatures/AttrP_y_QP.csv", ios::app );
  }
}

void closeFeaturesExtracting() {
  if ( OorGorA == 0 ) {
    GeomI_TOri_file.close();
    GeomI_Resi_file.close();
    GeomI_Occu_file.close();
    GeomI_y_QP_file.close();
    GeomP_TOri_file.close();
    GeomP_Resi_file.close();
    GeomP_Occu_file.close();
    GeomP_y_QP_file.close();
  } else if ( OorGorA > 0 ) {
    AttrI_TOri_file.close();
    AttrI_Resi_file.close();
    AttrI_Occu_file.close();
    AttrI_y_QP_file.close();
    AttrP_TOri_file.close();
    AttrP_Resi_file.close();
    AttrP_Occu_file.close();
    AttrP_y_QP_file.close();
  }
}

double nDecimalDouble( const double& dbNum, int n ) {
  int temp = dbNum * pow( 10, n );
  if ( ( (int)( dbNum * pow( 10, n + 1 ) ) - temp * 10 ) > 4 ) temp += 1;

  return (double)temp / pow( 10.0, n );
}

//void checkData( int width, int height, int frameNum ) {
//  for ( int i = 0; i < frameNum; i++ ) {
//    for ( int j = 0; j < height; j++ ) {
//      cout << "¡¾" << j << "¡¿: ";
//      for ( int k = 0; k < width; k++ ) { cout << occupancyData[i][j * mapWidth + k] - 0 << " "; }
//      cout << endl;
//    }
//    cout << "-========================-" << endl;
//  }
//}

#endif  // __MesksLibCPP__