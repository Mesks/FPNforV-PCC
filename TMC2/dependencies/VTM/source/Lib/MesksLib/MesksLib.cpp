#ifndef __MesksLibCPP__
#define __MesksLibCPP__

#include "MesksLib.h"

unsigned int*** occupancyData = NULL;
int			          picWidth = 0;
int			          picHeight = 0;
int                   mapWidth = 0;
int                   mapHeight = 0;
int                   mapFrameCount = 0;
int                   realtimeFrameCount = -1;
int                   OorGorA = -2;
bool* sideBlock_key = NULL;
double*** v_side = NULL;
double*** h_side = NULL;
string                CUkeys;
PyObject* pModule = NULL;
PyObject* pFunc_nnDriver = NULL;
double                FPNRuntime = 0;
double                postProcessRunTime = 0;

void initOccupancyData(int width, int height) {
    mapWidth = width;
    mapHeight = height;
    occupancyData = new unsigned int** [mapFrameCount];
    for (int i = 0; i < mapFrameCount; i++) {
        occupancyData[i] = new unsigned int* [mapWidth];
        for (int j = 0; j < mapWidth; j++) {
            // cout << "mapFrameCount: " << mapFrameCount << ", mapWidth: " << mapWidth << ", mapHeight: "<<mapHeight<<endl;
            occupancyData[i][j] = new unsigned int[mapHeight];
        }
    }
}

void deleteOccupancyData() {
    for (int i = 0; i < mapFrameCount; i++) {
        for (int j = 0; j < mapWidth; j++) {
            delete[] occupancyData[i][j];
            occupancyData[i][j] = NULL;
        }
        delete[] occupancyData[i];
        occupancyData[i] = NULL;
    }
    delete[] occupancyData;
    occupancyData = NULL;
}

void fillOccupancyData(int frameNum, vector<uint32_t> occupancyMap) {
    for (int i = 0; i < mapHeight; i++) {
        for (int j = 0; j < mapWidth; j++) {
            //cout << "i * mapWidth + j: " << i * mapWidth + j << ", occupancyMap.size: " << occupancyMap.size()<<endl;
            occupancyData[frameNum][j][i] = occupancyMap[i * mapWidth + j];
        }
    }
}

double nDecimalDouble(const double& dbNum, int n) {
    int temp = dbNum * pow(10, n);
    if (((int)(dbNum * pow(10, n + 1)) - temp * 10) > 4)
        temp += 1;

    return (double)temp / pow(10.0, n);
}

void deleteSideBlockKey() {
    if (sideBlock_key != NULL) {
        delete[] sideBlock_key;
        sideBlock_key = NULL;
    }
}

void initSideBlockKey() {
    deleteSideBlockKey();

    int blockNum = (int)(picWidth / 64) * (int)(picHeight / 64);
    sideBlock_key = new bool[blockNum];
    for (int i = 0; i < blockNum; i++)
        sideBlock_key[i] = false;
}

void deleteSideArray() {
    if (v_side != NULL) {
        for (int j = 0; j < (int)(picWidth / 64) * (int)(picHeight / 64); j++) {
            for (int i = 0; i < 15; i++) {
                delete[] v_side[j][i];
                v_side[j][i] = NULL;
            }
            delete[] v_side[j];
            v_side[j] = NULL;
        }
        delete[] v_side;
        v_side = NULL;
    }
    if (h_side != NULL) {
        for (int j = 0; j < (int)(picWidth / 64) * (int)(picHeight / 64); j++) {
            for (int i = 0; i < 16; i++) {
                delete[] h_side[j][i];
                h_side[j][i] = NULL;
            }
            delete[] h_side[j];
            h_side[j] = NULL;
        }
        delete[] h_side;
        h_side = NULL;
    }
}

void initSideArray() {
    initSideBlockKey();
    deleteSideArray();
    int blockNum = (int)(picWidth / 64) * (int)(picHeight / 64);

    v_side = new double** [blockNum];
    h_side = new double** [blockNum];

    for (int i = 0; i < blockNum; i++) {
        v_side[i] = new double* [15];
        h_side[i] = new double* [16];
    }

    for (int j = 0; j < blockNum; j++) {
        for (int i = 0; i < 16; i++) {
            if (i > 0) v_side[j][i - 1] = new double[16];
            h_side[j][i] = new double[15];
        }
    }
}

bool isSkipPartition(int width, int height, int X, int Y, int currMode)
{
    double th = 0.5;
    //cout << (X % 64) << "-" << (Y % 64) << "-" << width << "-" << height << "-"<<currMode << endl;
    //cout << "location: " << realtimeFrameCount << "-" << relativeX << "-" << relativeY << "-"<<relativeWidth<<"-"<<relativeHeight<<", currentMode: " << currMode
    //     << endl;
    //cout << " vSideCount_BT: " << vSideCount_BT << ", hSideCount_BT: " << hSideCount_BT
    //    << ", vSideCount_TT_L: " << vSideCount_TT_L << ", hSideCount_TT_T: " << hSideCount_TT_T
    //    << ", vSideCount_TT_R: " << vSideCount_TT_R << ", hSideCount_TT_B: " << hSideCount_TT_B << ", currMode: " << currMode << endl;
    switch (currMode) {
    case 7:   // ETM_SPLIT_QT
    {
        return false;
        int relativeWidth = width / 4;
        int relativeHeight = height / 4;
        int relativeX = (X % 64) / 4;
        int relativeY = (Y % 64) / 4;
        int globalSeq = (int)(X / 64) + (int)(Y / 64) * (int)(picWidth / 64);

        double vSideCount_BT = 0;
        double hSideCount_BT = 0;
        //cout << "vSideCount_BT's x: " << relativeX + relativeWidth / 2 - 1 << ", y: " << relativeY << ", step: " << relativeHeight << endl;
        for (int i = relativeY; relativeWidth > 1 && i < relativeY + relativeHeight; i++)
            if (v_side[globalSeq][relativeX + relativeWidth / 2 - 1][i] >= th)
                vSideCount_BT++;
            //vSideCount_BT += v_side[relativeX + relativeWidth / 2 - 1][i];
        //cout << "hSideCount_BT's x: " << relativeX << ", y: " << relativeY + relativeHeight / 2 - 1 << ", step: " << relativeWidth << endl;
        for (int j = relativeX; relativeHeight > 1 && j < relativeX + relativeWidth; j++)
            if (h_side[globalSeq][j][relativeY + relativeHeight / 2 - 1] >= th)
                hSideCount_BT++;
            //hSideCount_BT += h_side[j][relativeY + relativeHeight / 2 - 1];

        //if (vSideCount_BT == relativeHeight && hSideCount_BT == relativeWidth)
        if (vSideCount_BT >= relativeHeight / 2 || hSideCount_BT >= relativeWidth / 2)
        //if (vSideCount_BT / relativeHeight >= th && hSideCount_BT / relativeWidth >= th)
            return false;
        break;
    }
    case 8:   // ETM_SPLIT_BT_H
    {
        if (OorGorA>0)return false;
        //return false;
        int relativeWidth = width / 4;
        int relativeHeight = height / 4;
        int relativeX = (X % 64) / 4;
        int relativeY = (Y % 64) / 4;
        int globalSeq = (int)(X / 64) + (int)(Y / 64)* (int)(picWidth/64);

        double vSideCount_BT = 0;
        double hSideCount_BT = 0;

        for (int i = relativeY; i < relativeY + relativeHeight; i++)
            if (v_side[globalSeq][relativeX + relativeWidth / 2 - 1][i] >= th)
                vSideCount_BT++;
            //vSideCount_BT += v_side[relativeX + relativeWidth / 2 - 1][i];
        for (int j = relativeX; j < relativeX + relativeWidth; j++)
            if (h_side[globalSeq][j][relativeY + relativeHeight / 2 - 1] >= th)
                hSideCount_BT++;
            //hSideCount_BT += h_side[j][relativeY + relativeHeight / 2 - 1];

        //if (hSideCount_BT == relativeWidth /*&& vSideCount_BT != relativeHeight && !((width == 32 && height == 32 || width == 16 && height == 16))*/) {
        if (hSideCount_BT >= relativeWidth / 2) {
            //if (hSideCount_BT > 0) {
        //if (/*vSideCount_BT / relativeHeight >= th ||*/ hSideCount_BT / relativeWidth >= th){
            return false;
        }
        break;
    }
    case 9:   // ETM_SPLIT_BT_V
    {
        if (OorGorA>0)return false;
        //return false;
        int relativeWidth = width / 4;
        int relativeHeight = height / 4;
        int relativeX = (X % 64) / 4;
        int relativeY = (Y % 64) / 4;
        int globalSeq = (int)(X / 64) + (int)(Y / 64) * (int)(picWidth / 64);

        double vSideCount_BT = 0;
        double hSideCount_BT = 0;
        for (int i = relativeY; i < relativeY + relativeHeight; i++)
            if (v_side[globalSeq][relativeX + relativeWidth / 2 - 1][i] >= th)
                vSideCount_BT++;
            //vSideCount_BT += v_side[relativeX + relativeWidth / 2 - 1][i];
        for (int j = relativeX; j < relativeX + relativeWidth; j++)
            if (h_side[globalSeq][j][relativeY + relativeHeight / 2 - 1] >= th)
                hSideCount_BT++;
            //hSideCount_BT += h_side[j][relativeY + relativeHeight / 2 - 1];

        //if (vSideCount_BT == relativeHeight /*&&hSideCount_BT != relativeWidth&& !((width == 32&&height==32 || width == 16 && height == 16) )*/) {
        if (vSideCount_BT >= relativeHeight / 2) {
            //if (vSideCount_BT > 0) {
        //if (vSideCount_BT / relativeHeight >= th /*|| hSideCount_BT / relativeWidth >= th*/) {
            return false;
        }
        break;
    }
    case 10:   // ETM_SPLIT_TT_H
    {
        //return false;
        int relativeWidth = width / 4;
        int relativeHeight = height / 4;
        int relativeX = (X % 64) / 4;
        int relativeY = (Y % 64) / 4;
        int globalSeq = (int)(X / 64) + (int)(Y / 64) * (int)(picWidth / 64);
        if (height <= 4) return true;

        int hSideCount_BT = 0;
        for (int j = relativeX; relativeHeight > 1 && j < relativeX + relativeWidth; j++)
            if (h_side[globalSeq][j][relativeY + relativeHeight / 2 - 1] >= th)
                hSideCount_BT++;
            //hSideCount_BT += h_side[j][relativeY + relativeHeight / 2 - 1];

        double vSideCount_TT_L = 0;
        double hSideCount_TT_T = 0;
        double vSideCount_TT_R = 0;
        double hSideCount_TT_B = 0;
        for (int row = relativeX; relativeHeight > 3 && row < relativeX + relativeWidth; row++) {
            if (h_side[globalSeq][row][relativeY + relativeHeight / 4 - 1] >= th)hSideCount_TT_T++;
            if (h_side[globalSeq][row][relativeY + relativeHeight * 3 / 4 - 1] >= th)hSideCount_TT_B++;
            //hSideCount_TT_T += h_side[row][relativeY + relativeHeight / 4 - 1];
            //hSideCount_TT_B += h_side[row][relativeY + relativeHeight * 3 / 4 - 1];
        }

        if (hSideCount_BT < relativeWidth / 2 &&
            //(hSideCount_TT_T == relativeWidth && hSideCount_TT_B == relativeWidth))
            (hSideCount_TT_T >= relativeWidth / 2 && hSideCount_TT_B >= relativeWidth / 2 ))
        //if (hSideCount_BT / relativeWidth < th &&
        //    (hSideCount_TT_T / relativeWidth >= th && hSideCount_TT_B / relativeWidth >= th))
            return false;
        break;
    }
    case 11:   // ETM_SPLIT_TT_V
    {
        //return false;
        int relativeWidth = width / 4;
        int relativeHeight = height / 4;
        int relativeX = (X % 64) / 4;
        int relativeY = (Y % 64) / 4;
        int globalSeq = (int)(X / 64) + (int)(Y / 64) * (int)(picWidth / 64);
        if (width <= 4) return true;

        int vSideCount_BT = 0;
        int hSideCount_BT = 0;
        for (int i = relativeY; relativeWidth > 1 && i < relativeY + relativeHeight; i++)
            if (v_side[globalSeq][relativeX + relativeWidth / 2 - 1][i] >= th)
                vSideCount_BT++;
            //vSideCount_BT += v_side[relativeX + relativeWidth / 2 - 1][i];

        double vSideCount_TT_L = 0;
        double hSideCount_TT_T = 0;
        double vSideCount_TT_R = 0;
        double hSideCount_TT_B = 0;
        for (int col = relativeY; relativeWidth > 3 && col < relativeY + relativeHeight; col++) {
            if (v_side[globalSeq][relativeX + relativeWidth / 4 - 1][col] >= th)vSideCount_TT_L++;
            if (v_side[globalSeq][relativeX + relativeWidth * 3 / 4 - 1][col] >= th)vSideCount_TT_R++;
            //vSideCount_TT_L += v_side[relativeX + relativeWidth / 4 - 1][col];
            //vSideCount_TT_R += v_side[relativeX + relativeWidth * 3 / 4 - 1][col];
        }

        if (vSideCount_BT < relativeHeight / 2 &&
            //(vSideCount_TT_L == relativeHeight && vSideCount_TT_R == relativeHeight ))
            (vSideCount_TT_L >= relativeHeight / 2 && vSideCount_TT_R >= relativeHeight / 2))
        //if (vSideCount_BT / relativeHeight < th &&
        //        (vSideCount_TT_L / relativeHeight >= th && vSideCount_TT_R / relativeHeight >= th))
            return false;
        break;
    }
    }
    return true;
}

void MCCNNInit()
{
    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('../external')");
    
    // Calling array_tutorial from mymodule
    pModule = PyImport_ImportModule("FPN_driver");
    if (!pModule)
        cout << "Python module is not import, please check the file exists and no compilation errors." << endl;
    pFunc_nnDriver = PyObject_GetAttrString(pModule, "nnDriver");
    if (pModule && !pFunc_nnDriver)
        cout << "Python module is appear but function is invalid, please check no file with the same name (not including "
        "file type)."
        << endl;

    PyObject* pArgs = PyTuple_New(2);
    PyTuple_SetItem(pArgs, 0, Py_BuildValue("s", "../external/FPN_GeomP.pt"));
    PyTuple_SetItem(pArgs, 1, Py_BuildValue("s", "../external/FPN_AttrI.pt"));

    PyObject* pFunc_model_init = PyObject_GetAttrString(pModule, "model_init");
    PyEval_CallObject(pFunc_model_init, pArgs);
    Py_DECREF(pFunc_model_init);
    Py_DECREF(pArgs);
}

void MCCNNDestroy() {
    Py_DECREF(pModule);
    Py_DECREF(pFunc_nnDriver);
    Py_Finalize();
}

int nnPredict(const CodingStructure* bestCS, Partitioner* partitioner, int modelType)
{
    int cuX = bestCS->area.Y().lumaPos().x;
    int cuY = bestCS->area.Y().lumaPos().y;
    int globalSeq = (int)(cuX / 64) + (int)(cuY / 64) * (int)(picWidth / 64);

    // Build the input. We use list instead of numpy to avoid "import_array", to many times invoke attributes lots of time consumption.
    // Note that the tensor of pytorch is [N, C, H, W], so put the column to the first dim, it means here happens a transfer.
    PyObject* pArgs, * pReturn;
    PyObject* list_BaseFeatures,* list_OccuFeatures, * list_arg;
    list_BaseFeatures = PyList_New(0);
    list_OccuFeatures = PyList_New(0);

    for (int i = 0; i < 64; i++)
    {
        PyObject* PyList_Base = PyList_New(64);
        PyObject* PyList_Occu = PyList_New(64);
        for (int j = 0; j < 64; j++)
        {
            if (modelType == 0 || modelType == 2) {
                PyList_SetItem(
                    PyList_Base, j,
                    PyFloat_FromDouble(nDecimalDouble((double)bestCS->getTrueOrgBuf().Y().at(i, j) / 4.0 / 255.0, 3)));
                PyList_SetItem(PyList_Occu, j,
                    PyFloat_FromDouble((int)(occupancyData[int(realtimeFrameCount / 2)][i + cuX][j + cuY])));
            }
            else if (modelType == 1 || modelType == 3) {
                PyList_SetItem(
                    PyList_Base, j,
                    //PyFloat_FromDouble(nDecimalDouble((double)bestCS->getPredBuf(partitioner->currArea().block(COMPONENT_Y)).at(i, j) / 4.0 / 255.0, 3)));
                    PyFloat_FromDouble(nDecimalDouble((abs((double) bestCS->getPredBuf(partitioner->currArea().block(COMPONENT_Y)).at(i, j)
                -(double)bestCS->getTrueOrgBuf().Y().at(i, j))
                / 4.0) / 255.0, 3)));
                PyList_SetItem(PyList_Occu, j,
                    PyFloat_FromDouble((int)(occupancyData[int(realtimeFrameCount / 2)][i + cuX][j + cuY])));
            }
        }
        PyList_Append(list_BaseFeatures, PyList_Base);
        PyList_Append(list_OccuFeatures, PyList_Occu);
        Py_DECREF(PyList_Base);
        Py_DECREF(PyList_Occu);
    }

    pArgs = PyTuple_New(4);
    PyTuple_SetItem(pArgs, 0, list_BaseFeatures);
    PyTuple_SetItem(pArgs, 1, list_OccuFeatures);
    PyTuple_SetItem(pArgs, 2, Py_BuildValue("d", nDecimalDouble(bestCS->baseQP / 51.0, 3)));
    PyTuple_SetItem(pArgs, 3, Py_BuildValue("i", modelType));

    pReturn = PyEval_CallObject(pFunc_nnDriver, pArgs);

    int SizeOfList = PyList_Size(pReturn);
    for (int dim = 0; dim < SizeOfList; dim++)
    {
        PyObject* ListItem = PyList_GetItem(pReturn, dim);
        if (dim == 0) {
            for (int i = 0; i < 15; i++)
            {
                for (int j = 0; j < 16; j++)
                {
                    v_side[globalSeq][i][j] = PyFloat_AsDouble(PyList_GetItem(ListItem, i * 16 + j));
                    h_side[globalSeq][j][i] = PyFloat_AsDouble(PyList_GetItem(ListItem, 240 + i * 16 + j));
                }
            }
            sideBlock_key[globalSeq] = true;
        }
        //else FPNRuntime += PyFloat_AsDouble(PyList_GetItem(ListItem));
    }

    //checkSide(bestCS->area.Y().lumaPos().x, bestCS->area.Y().lumaPos().y);

    Py_DECREF(pArgs);
    Py_DECREF(pReturn);
    Py_DECREF(list_BaseFeatures);
    Py_DECREF(list_OccuFeatures);

    return 0;
}

void checkSide()
{
    double xx = v_side[0][0][0];
    if (v_side[0][0] == NULL) {
        cout << "MesksDone!!!" << endl;
    }
    else {
        cout << xx << endl;
    }
    exit(0);
    cout << "prediction output v_side: " << endl;
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 15; j++)
            cout << v_side[j][i] << ",";
        cout << endl;
    }
    cout << endl << "prediction output h_side: " << endl;
    for (int i = 0; i < 15; i++)
    {
        for (int j = 0; j < 16; j++)
            cout << h_side[j][i] << ",";
        cout << endl;
    }
}

void checkSide(int X, int Y)
{
    cout << realtimeFrameCount << "-" << X << "-" << Y << " prediction output v_side: " << endl;
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 15; j++)
            cout << v_side[j][i] << ",";
        cout << endl;
    }
    cout << endl << realtimeFrameCount << "-" << X << "-" << Y << " prediction output h_side: " << endl;
    for (int i = 0; i < 15; i++)
    {
        for (int j = 0; j < 16; j++)
            cout << h_side[j][i] << ",";
        cout << endl;
    }
}

#endif  // __MesksLibCPP__