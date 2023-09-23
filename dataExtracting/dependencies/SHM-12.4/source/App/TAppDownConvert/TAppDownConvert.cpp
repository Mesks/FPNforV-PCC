/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.  
 *
 * Copyright (c) 2010-2016, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/** \file     TAppDownConvert.cpp
    \brief    Down convert application main
*/

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "DownConvert.h"

typedef struct
{
  int            stride;
  int            lines;
  short* data;
  short* data2;
} ColorComponent;

typedef struct
{
  int inputBitDepth;
  int outputBitDepth;
  ColorComponent y;
  ColorComponent u;
  ColorComponent v;
} YuvFrame;


void
  createColorComponent( ColorComponent& c, int maxwidth, int maxheight )
{
  maxwidth  = ( ( maxwidth  + 15 ) >> 4 ) << 4;
  maxheight = ( ( maxheight + 15 ) >> 4 ) << 4;
  int size  = maxwidth * maxheight;
  c.stride  = maxwidth;
  c.lines   = maxheight;
  c.data    = new short [ size ];
  c.data2   = new short [ size ];

  if( ! c.data || ! c.data2 )
  {
    fprintf(stderr, "\nERROR: memory allocation failed!\n\n");
    exit(1);
  }
}

void
  deleteColorComponent( ColorComponent& c )
{
  delete[] c.data;
  delete[] c.data2;
  c.stride  = 0;
  c.lines   = 0;
  c.data    = 0;
  c.data2   = 0;
}

int
  readColorComponent( ColorComponent& c, FILE* file, int width, int height, int inputBitDepth, bool second )
{
  assert( width  <= c.stride );
  assert( height <= c.lines  );
  unsigned char *temp=NULL;
  
  int iMaxPadWidth  = gMin( c.stride, ( ( width  + 15 ) >> 4 ) << 4 );
  int iMaxPadHeight = gMin( c.lines,  ( ( height + 31 ) >> 5 ) << 5 );

  for( int i = 0; i < height; i++ )
  {
    short* buffer = ( second ? c.data2 : c.data ) + i * c.stride;
    int            rsize;
    
    if (inputBitDepth == 10)
    {
       rsize = (int)fread( buffer, sizeof(short), width, file );
    }
    else
    {
       if (temp == NULL)
         temp = (unsigned char * )calloc(width, sizeof(unsigned char));
       rsize = (int)fread( temp, sizeof(unsigned char), width, file );
       for (int j = 0; j < width; j++)
         buffer[j] = (short) temp[j];
    }
    if( rsize != width )
    {
      return 1;
    }
    for( int xp = width; xp < iMaxPadWidth; xp++ )
    {
      buffer[xp] = buffer[xp-1];
    }
  }
  for( int yp = height; yp < iMaxPadHeight; yp++ )
  {
    short* buffer  = ( second ? c.data2 : c.data ) + yp * c.stride;
    short* bufferX = buffer - c.stride;
    for( int xp = 0; xp < c.stride; xp++ )
    {
      buffer[xp] = bufferX[xp];
    }
  }
  if (temp!=NULL)
    free(temp);
  return 0;
}

void
  duplicateColorComponent( ColorComponent& c )
{
  memcpy( c.data2, c.data, c.stride * c.lines * sizeof(short) );
}

void
  combineTopAndBottomInColorComponent( ColorComponent& c, Bool bBotField )
{
  int            offs = ( bBotField ? c.stride : 0 );
  short* pDes = c.data  + offs;
  short* pSrc = c.data2 + offs;
  for( int i = 0; i < c.lines / 2; i++, pDes += 2*c.stride, pSrc += 2*c.stride )
  {
    memcpy( pDes, pSrc, c.stride * sizeof(short) );
  }
}

void
  writeColorComponent( ColorComponent& c, FILE* file, int width, int height, int outputBitDepth, bool second )
{
  assert( width  <= c.stride );
  assert( height <= c.lines  );
  unsigned char* temp = NULL;
       
  for( int i = 0; i < height; i++ )
  {
    short* buffer = ( second ? c.data2 : c.data ) + i * c.stride;
    int            wsize;
    
    if ( outputBitDepth == 8 )
    {
      if (temp == NULL)
        temp = (unsigned char * )calloc(width, sizeof(unsigned char));
      for (int j = 0; j < width; j ++)
        temp[j] = (unsigned char)buffer[j];
      wsize = (int)fwrite( temp, sizeof(unsigned char), width, file );
    }
    else
    {
      wsize = (int)fwrite( buffer, sizeof(short), width, file );
    }

    if( wsize != width )
    {
      fprintf(stderr, "\nERROR: while writing to output file!\n\n");
      exit(1);
    }
  }
}


void
  createFrame( YuvFrame& f, int width, int height, int inputBitDepth, int outputBitDepth )
{
  f.inputBitDepth = inputBitDepth;
  f.outputBitDepth = outputBitDepth;
  createColorComponent( f.y, width,      height      );
  createColorComponent( f.u, width >> 1, height >> 1 );
  createColorComponent( f.v, width >> 1, height >> 1 );
}

void
  deleteFrame( YuvFrame& f )
{
  deleteColorComponent( f.y );
  deleteColorComponent( f.u );
  deleteColorComponent( f.v );
}

int
  readFrame( YuvFrame& f, FILE* file, int width, int height, bool second = false )
{
  ROTRS( readColorComponent( f.y, file, width,      height,      f.inputBitDepth, second ), 1 );
  ROTRS( readColorComponent( f.u, file, width >> 1, height >> 1, f.inputBitDepth, second ), 1 );
  ROTRS( readColorComponent( f.v, file, width >> 1, height >> 1, f.inputBitDepth, second ), 1 );
  return 0;
}

void
  duplicateFrame( YuvFrame& f )
{
  duplicateColorComponent( f.y );
  duplicateColorComponent( f.u );
  duplicateColorComponent( f.v );
}

void
  combineTopAndBottomInFrame( YuvFrame& f, Bool botField )
{
  combineTopAndBottomInColorComponent( f.y, botField );
  combineTopAndBottomInColorComponent( f.u, botField );
  combineTopAndBottomInColorComponent( f.v, botField );
}

void
  writeFrame( YuvFrame& f, FILE* file, int width, int height, bool both = false )
{
  writeColorComponent( f.y, file, width,      height,      f.outputBitDepth, false );
  writeColorComponent( f.u, file, width >> 1, height >> 1, f.outputBitDepth, false );
  writeColorComponent( f.v, file, width >> 1, height >> 1, f.outputBitDepth, false );

  if( both )
  {
    writeColorComponent( f.y, file, width,      height,      f.outputBitDepth, true );
    writeColorComponent( f.u, file, width >> 1, height >> 1, f.outputBitDepth, true );
    writeColorComponent( f.v, file, width >> 1, height >> 1, f.outputBitDepth, true );
  }
}


void
  print_usage_and_exit( int test, const char* name, const char* message = 0 )
{
  if( test )
  {
    if( message )
    {
      fprintf ( stderr, "\nERROR: %s\n", message );
    }
    fprintf (   stderr, "\nUsage: %s <win> <hin> <in> <wout> <hout> <out> [<bin> <bout> <method> <t> [<skip> [<frms>]]] [[-phase <args>] ]\n\n", name );
    fprintf (   stderr, "  win     : input width  (luma samples)\n" );
    fprintf (   stderr, "  hin     : input height (luma samples)\n" );
    fprintf (   stderr, "  in      : input file\n" );
    fprintf (   stderr, "  wout    : output width  (luma samples)\n" );
    fprintf (   stderr, "  hout    : output height (luma samples)\n" );
    fprintf (   stderr, "  out     : output file\n" );
    fprintf (   stderr, "\n--------------------------- OPTIONAL ---------------------------\n\n" );
    fprintf (   stderr, "  bin     : input bit depth (8 or 10) (default: 8)\n" );
    fprintf (   stderr, "  bout    : output bit depth (8 or 10) (default: 8)\n" );
    fprintf (   stderr, "  method  : sampling method (0, 1, 2, 3, 4) (default: 0)\n" );
    fprintf (   stderr, "  t       : number of temporal downsampling stages (default: 0)\n" );
    fprintf (   stderr, "  skip    : number of frames to skip at start (default: 0)\n" );
    fprintf (   stderr, "  frms    : number of frames wanted in output file (default: max)\n" );
    fprintf (   stderr, "\n-------------------------- OVERLOADED --------------------------\n\n" );
    fprintf (   stderr, " -phase <in_uv_ph_x> <in_uv_ph_y> <out_uv_ph_x> <out_uv_ph_y>\n");
    fprintf (   stderr, "   in_uv_ph_x : input  chroma phase shift in horizontal direction (default:-1)\n" );
    fprintf (   stderr, "   in_uv_ph_y : input  chroma phase shift in vertical   direction (default: 0)\n" );
    fprintf (   stderr, "   out_uv_ph_x: output chroma phase shift in horizontal direction (default:-1)\n" );
    fprintf (   stderr, "   out_uv_ph_y: output chroma phase shift in vertical   direction (default: 0)\n" );
    fprintf (   stderr, "\n-------------------------- EXAMPLE --------------------------\n\n" );
    fprintf (   stderr, "\nUsage: %s 4096 2048 input_video_4096x2048_10bits.yuv 2048 1024 output_video_2048x1024_8bits.yuv 10 8\n\n", name );
    fprintf (   stderr, "\n\n");
    exit    (   1 );
  }
}


void
  updateCropParametersFromFile( ResizeParameters& cRP, FILE* cropFile, int resamplingMethod, char* name )
{
  int crop_x0 = 0;
  int crop_y0 = 0;
  int crop_w  = 0;
  int crop_h  = 0;
  if( fscanf( cropFile, "%d,%d,%d,%d\n", &crop_x0, &crop_y0, &crop_w, &crop_h ) == 4 )
  {
    cRP.m_iLeftFrmOffset      = crop_x0;
    cRP.m_iTopFrmOffset       = crop_y0;
    cRP.m_iScaledRefFrmWidth  = crop_w;
    cRP.m_iScaledRefFrmHeight = crop_h;
  }
  print_usage_and_exit( cRP.m_iLeftFrmOffset     & 1 || cRP.m_iTopFrmOffset       & 1,                                              name, "cropping parameters must be even values" );
  print_usage_and_exit( cRP.m_iScaledRefFrmWidth & 1 || cRP.m_iScaledRefFrmHeight & 1,                                              name, "cropping parameters must be even values" );
  print_usage_and_exit( resamplingMethod == 2 && cRP.m_iScaledRefFrmWidth  != gMin( cRP.m_iRefLayerFrmWidth,  cRP.m_iFrameWidth  ),  name, "crop dimensions must be the same as the minimal dimensions" );
  print_usage_and_exit( resamplingMethod == 2 && cRP.m_iScaledRefFrmHeight != gMin( cRP.m_iRefLayerFrmHeight, cRP.m_iFrameHeight ),  name, "crop dimensions must be the same as the minimal dimensions" );
  print_usage_and_exit( cRP.m_iScaledRefFrmWidth  > gMax( cRP.m_iRefLayerFrmWidth,  cRP.m_iFrameWidth  ),                            name, "wrong crop window size" );
  print_usage_and_exit( cRP.m_iScaledRefFrmHeight > gMax( cRP.m_iRefLayerFrmHeight, cRP.m_iFrameHeight ),                            name, "wrong crop window size" );
  print_usage_and_exit( cRP.m_iScaledRefFrmWidth  < gMin( cRP.m_iRefLayerFrmWidth,  cRP.m_iFrameWidth  ),                            name, "wrong crop window size" );
  print_usage_and_exit( cRP.m_iScaledRefFrmHeight < gMin( cRP.m_iRefLayerFrmHeight, cRP.m_iFrameHeight ),                            name, "wrong crop window size" );
  print_usage_and_exit( cRP.m_iLeftFrmOffset + cRP.m_iScaledRefFrmWidth  > gMax( cRP.m_iRefLayerFrmWidth,  cRP.m_iFrameWidth  ),     name, "wrong crop window size and origin" );
  print_usage_and_exit( cRP.m_iTopFrmOffset  + cRP.m_iScaledRefFrmHeight > gMax( cRP.m_iRefLayerFrmHeight, cRP.m_iFrameHeight ),     name, "wrong crop window size and origin" );
}


void
  resampleFrame( YuvFrame&          rcFrame,
  DownConvert&       rcDownConvert,
  ResizeParameters&  rcRP,
  int                resamplingMethod,
  int                resamplingMode,
  bool               resampling,
  bool               upsampling,
  bool               bSecondInputFrame )
{
  assert( upsampling == 0 );

  //===== downsampling =====
  ResizeParameters cRP = rcRP;
  {
    Int iRefVerMbShift        = ( cRP.m_bRefLayerFrameMbsOnlyFlag ? 4 : 5 );
    Int iScaledVerShift       = ( cRP.m_bFrameMbsOnlyFlag         ? 1 : 2 );
    Int iHorDiv               = ( cRP.m_iFrameWidth    <<               1 );
    Int iVerDiv               = ( cRP.m_iFrameHeight   << iScaledVerShift );
    Int iRefFrmW              = ( ( cRP.m_iFrameWidth   + ( 1 <<               4 ) - 1 ) >>               4 ) <<               4;        // round to next multiple of 16
    Int iRefFrmH              = ( ( cRP.m_iFrameHeight  + ( 1 <<  iRefVerMbShift ) - 1 ) >>  iRefVerMbShift ) <<  iRefVerMbShift;        // round to next multiple of 16 or 32 (for interlaced)
    Int iScaledRefFrmW        = ( ( cRP.m_iScaledRefFrmWidth  * iRefFrmW + ( iHorDiv >> 1 ) ) / iHorDiv ) <<               1;  // scale and round to next multiple of  2
    Int iScaledRefFrmH        = ( ( cRP.m_iScaledRefFrmHeight * iRefFrmH + ( iVerDiv >> 1 ) ) / iVerDiv ) << iScaledVerShift;  // scale and round to next multiple of  2 or  4 (for interlaced)
    cRP.m_iFrameWidth         = iRefFrmW;
    cRP.m_iFrameHeight        = iRefFrmH;
    cRP.m_iScaledRefFrmWidth  = iScaledRefFrmW;
    cRP.m_iScaledRefFrmHeight = iScaledRefFrmH;
    cRP.inputBitDepth         = rcFrame.inputBitDepth;
    cRP.outputBitDepth        = rcFrame.outputBitDepth;
  }
  assert( resamplingMethod == 0 );
  if( resamplingMode < 4 )
  {
    rcDownConvert.downsamplingSVC       ( rcFrame.y.data,  rcFrame.y.stride, rcFrame.u.data,  rcFrame.u.stride, rcFrame.v.data,  rcFrame.v.stride, &cRP, resamplingMode == 3 );
    return;
  }
}



int
  main( int argc, char *argv[] )
{
  //===== set standard resize parameters =====
  ResizeParameters cRP;
  cRP.m_bRefLayerFrameMbsOnlyFlag   = true;
  cRP.m_bFrameMbsOnlyFlag           = true;
  cRP.m_bRefLayerFieldPicFlag       = false;
  cRP.m_bFieldPicFlag               = false;
  cRP.m_bRefLayerBotFieldFlag       = false;
  cRP.m_bBotFieldFlag               = false;
  cRP.m_bRefLayerIsMbAffFrame       = false;
  cRP.m_bIsMbAffFrame               = false;
#if ZERO_PHASE
  cRP.m_iRefLayerChromaPhaseX       = 0;
  cRP.m_iRefLayerChromaPhaseY       = 1;
  cRP.m_iChromaPhaseX               = 0;
  cRP.m_iChromaPhaseY               = 1;
#else
  cRP.m_iRefLayerChromaPhaseX       = -1;
  cRP.m_iRefLayerChromaPhaseY       = 0;
  cRP.m_iChromaPhaseX               = -1;
  cRP.m_iChromaPhaseY               = 0;
#endif
  cRP.m_iRefLayerFrmWidth           = 0;
  cRP.m_iRefLayerFrmHeight          = 0;
  cRP.m_iScaledRefFrmWidth          = 0;
  cRP.m_iScaledRefFrmHeight         = 0;
  cRP.m_iFrameWidth                 = 0;
  cRP.m_iFrameHeight                = 0;
  cRP.m_iLeftFrmOffset              = 0;
  cRP.m_iTopFrmOffset               = 0;
  //cRP.m_iExtendedSpatialScalability = 0;
  cRP.m_iLevelIdc                   = 0;

  //===== init parameters =====
  FILE* inputFile                   = 0;
  FILE* outputFile                  = 0;
  FILE* croppingParametersFile      = 0;
  int   inputBitDepth               = 8;
  int   outputBitDepth              = 8;
  int   resamplingMethod            = 0;
  int   resamplingMode              = 0;
  bool  croppingInitialized         = false;
  bool  phaseInitialized            = false;
  bool  methodInitialized           = false;
  bool  resampling                  = false;
  bool  upsampling                  = false;
  int   numSpatialDyadicStages      = 0;
  int   skipBetween                 = 0;
  int   skipAtStart                 = 0;
  int   maxNumOutputFrames          = 0;


  //===== read input parameters =====
  print_usage_and_exit( ( argc < 7 || argc > 24 ), argv[0], "wrong number of arguments" );
  cRP.m_iRefLayerFrmWidth   = atoi  ( argv[1] );
  cRP.m_iRefLayerFrmHeight  = atoi  ( argv[2] );
  inputFile                 = fopen ( argv[3], "rb" );
  cRP.m_iFrameWidth         = atoi  ( argv[4] );
  cRP.m_iFrameHeight        = atoi  ( argv[5] );
  outputFile                = fopen ( argv[6], "wb" );
  print_usage_and_exit( ! inputFile,  argv[0], "failed to open input file" );
  print_usage_and_exit( ! outputFile, argv[0], "failed to open input file" );
  print_usage_and_exit( cRP.m_iRefLayerFrmWidth > cRP.m_iFrameWidth && cRP.m_iRefLayerFrmHeight < cRP.m_iFrameHeight, argv[0], "mixed upsampling and downsampling not supported" );
  print_usage_and_exit( cRP.m_iRefLayerFrmWidth < cRP.m_iFrameWidth && cRP.m_iRefLayerFrmHeight > cRP.m_iFrameHeight, argv[0], "mixed upsampling and downsampling not supported" );
  for( int i = 7; i < argc; )
  {
    if( ! strcmp( argv[i], "-phase" ) )
    {
      print_usage_and_exit( resamplingMethod != 0,          argv[0], "phases only supported in normative resampling" );
      print_usage_and_exit( phaseInitialized || argc < i+5, argv[0], "wrong number of phase parameters" );
      phaseInitialized = true;
      i++;
      cRP.m_iRefLayerChromaPhaseX = atoi( argv[i++] );
      cRP.m_iRefLayerChromaPhaseY = atoi( argv[i++] );
      cRP.m_iChromaPhaseX         = atoi( argv[i++] );
      cRP.m_iChromaPhaseY         = atoi( argv[i++] );
      print_usage_and_exit( cRP.m_iRefLayerChromaPhaseX > 0 || cRP.m_iRefLayerChromaPhaseX < -1, argv[0], "wrong phase x parameters (range : [-1, 0])");
      print_usage_and_exit( cRP.m_iRefLayerChromaPhaseY > 1 || cRP.m_iRefLayerChromaPhaseY < -1, argv[0], "wrong phase x parameters (range : [-1, 1])");
      print_usage_and_exit( cRP.m_iChromaPhaseX         > 0 || cRP.m_iChromaPhaseX         < -1, argv[0], "wrong phase x parameters (range : [-1, 0])");
      print_usage_and_exit( cRP.m_iChromaPhaseY         > 1 || cRP.m_iChromaPhaseY         < -1, argv[0], "wrong phase x parameters (range : [-1, 1])");
    }
    else if (i == 7) // input bit depth
    {
      inputBitDepth = atoi( argv[i++] );
      print_usage_and_exit( inputBitDepth != 8 && inputBitDepth != 10, argv[0], "wrong input bit depth (8 bit or 10 bit)");
    }
    else if (i == 8) // output bit depth
    {
      outputBitDepth = atoi( argv[i++] );
      print_usage_and_exit( outputBitDepth != 8 && outputBitDepth != 10, argv[0], "wrong output bit depth (8 bit or 10 bit)");
    }
    else if (i == 9) // downsampling methods
    {
      methodInitialized = true;
      resamplingMethod  = atoi( argv[i++] );
      print_usage_and_exit( resamplingMethod < 0 || resamplingMethod > 4, argv[0], "unsupported method" );
      if( resamplingMethod > 2 )
      {
        print_usage_and_exit( cRP.m_iRefLayerFrmWidth  > cRP.m_iFrameWidth,  argv[0], "method 3 and 4 are not supported for downsampling" );
        print_usage_and_exit( cRP.m_iRefLayerFrmHeight > cRP.m_iFrameHeight, argv[0], "method 3 and 4 are not supported for downsampling" );
      }
      if( resamplingMethod != 2 )
      {
        resampling  = true;
        upsampling  = ( cRP.m_iRefLayerFrmWidth < cRP.m_iFrameWidth ) || ( cRP.m_iRefLayerFrmHeight < cRP.m_iFrameHeight );
      }
      if( resamplingMethod == 1 )
      {
        if( upsampling )
        {
          int      div  = cRP.m_iFrameWidth / cRP.m_iRefLayerFrmWidth;
          if     ( div == 1) numSpatialDyadicStages =  0;
          else if( div == 2) numSpatialDyadicStages =  1;
          else if( div == 4) numSpatialDyadicStages =  2;
          else if( div == 8) numSpatialDyadicStages =  3;
          else               numSpatialDyadicStages = -1;
          print_usage_and_exit( numSpatialDyadicStages < 0,                           argv[0], "ratio not supported for dyadic upsampling method" );
          print_usage_and_exit( div * cRP.m_iRefLayerFrmWidth  != cRP.m_iFrameWidth,  argv[0], "ratio is not dyadic in dyadic mode" );
          print_usage_and_exit( div * cRP.m_iRefLayerFrmHeight != cRP.m_iFrameHeight, argv[0], "different horizontal and vertical ratio in dyadic mode" );
        }
        else
        {
          int      div  = cRP.m_iRefLayerFrmWidth / cRP.m_iFrameWidth;
          if     ( div == 1) numSpatialDyadicStages =  0;
          else if( div == 2) numSpatialDyadicStages =  1;
          else if( div == 4) numSpatialDyadicStages =  2;
          else if( div == 8) numSpatialDyadicStages =  3;
          else               numSpatialDyadicStages = -1;
          print_usage_and_exit( numSpatialDyadicStages < 0,                           argv[0], "ratio not supported for dyadic downsampling method" );
          print_usage_and_exit( div * cRP.m_iFrameWidth  != cRP.m_iRefLayerFrmWidth,  argv[0], "ratio is not dyadic in dyadic mode" );
          print_usage_and_exit( div * cRP.m_iFrameHeight != cRP.m_iRefLayerFrmHeight, argv[0], "different horizontal and vertical ratio in dyadic mode" );
        }
      }
    }
    else if( i == 10 ) // temporal stages
    {
      int TStages = atoi( argv[i++] );
      skipBetween = ( 1 << TStages ) - 1;
      print_usage_and_exit( TStages < 0,              argv[0], "negative number of temporal stages" );
    }
    else if( i == 11 ) // skipped frames
    {
      skipAtStart = atoi( argv[i++] );
      print_usage_and_exit( skipAtStart < 0,          argv[0], "negative number of skipped frames at start" );
    }
    else if( i == 12 ) // output frames
    {
      maxNumOutputFrames = atoi( argv[i++] );
      print_usage_and_exit( maxNumOutputFrames < 0 ,  argv[0], "negative number of output frames" );
    }
    else
    {
      print_usage_and_exit( true, argv[0], "error in command line parameters" );
    }
  }
  if( ! methodInitialized )
  {
    resampling  = true;
    upsampling  = ( cRP.m_iRefLayerFrmWidth < cRP.m_iFrameWidth ) || ( cRP.m_iRefLayerFrmHeight < cRP.m_iFrameHeight );
  }
  if( ! croppingInitialized )
  {
    if( resamplingMethod == 2 )
    {
      cRP.m_iScaledRefFrmWidth  = gMin( cRP.m_iRefLayerFrmWidth,  cRP.m_iFrameWidth  );
      cRP.m_iScaledRefFrmHeight = gMin( cRP.m_iRefLayerFrmHeight, cRP.m_iFrameHeight );
    }
    else
    {
      cRP.m_iScaledRefFrmWidth  = gMax( cRP.m_iRefLayerFrmWidth,  cRP.m_iFrameWidth  );
      cRP.m_iScaledRefFrmHeight = gMax( cRP.m_iRefLayerFrmHeight, cRP.m_iFrameHeight );
    }
  }

  //===== set basic parameters for resampling control =====
  if( resamplingMethod == 0 )
  {
    if( resamplingMode == 1 )
    {
      cRP.m_bRefLayerFrameMbsOnlyFlag = false;
      cRP.m_bFrameMbsOnlyFlag         = false;
    }
    else if( resamplingMode == 2 || resamplingMode == 3 )
    {
      cRP.m_bFrameMbsOnlyFlag     = false;
      if( ! upsampling )
      {
        cRP.m_bFieldPicFlag       = true;
        cRP.m_bBotFieldFlag       = ( resamplingMode == 3 );
      }
    }
    else if( resamplingMode == 4 || resamplingMode == 5 )
    {
      cRP.m_bRefLayerFrameMbsOnlyFlag = false;
      cRP.m_bRefLayerFieldPicFlag     = true;
    }
  }

  //===== initialize classes =====
  YuvFrame    cFrame;
  DownConvert cDownConvert;
  {
    int maxWidth  = gMax( cRP.m_iRefLayerFrmWidth,  cRP.m_iFrameWidth  );
    int maxHeight = gMax( cRP.m_iRefLayerFrmHeight, cRP.m_iFrameHeight );
    int minWidth  = gMin( cRP.m_iRefLayerFrmWidth,  cRP.m_iFrameWidth  );
    int minHeight = gMin( cRP.m_iRefLayerFrmHeight, cRP.m_iFrameHeight );
    int minWRnd16 = ( ( minWidth  + 15 ) >> 4 ) << 4;
    int minHRnd32 = ( ( minHeight + 31 ) >> 5 ) << 5;
    maxWidth      = ( ( maxWidth  * minWRnd16 + ( minWidth  << 4 ) - 1 ) / ( minWidth  << 4 ) ) << 4;
    maxHeight     = ( ( maxHeight * minHRnd32 + ( minHeight << 4 ) - 1 ) / ( minHeight << 4 ) ) << 4;
    createFrame( cFrame, maxWidth, maxHeight, inputBitDepth, outputBitDepth );
    cDownConvert.init(   maxWidth, maxHeight );
  }

  printf("Resampler\n\n");

  //===== loop over frames =====
  int   skip              = skipAtStart;
  int   writtenFrames     = 0;
  int   numInputFrames    = ( resamplingMode >= 4 && ! upsampling ? 2 : 1 );
  int   numOutputFrames   = ( resamplingMode >= 4 &&   upsampling ? 2 : 1 );
  bool  bFinished         = false;
  long  startTime         = clock();
  while( ! bFinished )
  {
    for( int inputFrame = 0; inputFrame < numInputFrames && ! bFinished; inputFrame++ )
    {
      //===== read input frame =====
      for( int numToRead = skip + 1; numToRead > 0 && ! bFinished; numToRead-- )
      {
        bFinished = ( readFrame( cFrame, inputFile, cRP.m_iRefLayerFrmWidth, cRP.m_iRefLayerFrmHeight, inputFrame != 0 ) != 0 );
      }
      skip = skipBetween;
      if( cRP.m_iExtendedSpatialScalability == 2 && ! bFinished )
      {
        updateCropParametersFromFile( cRP, croppingParametersFile, resamplingMethod, argv[0] );
      }

      //===== set resampling parameter =====
      if( resamplingMethod != 0 &&
        cRP.m_iScaledRefFrmWidth  == gMin( cRP.m_iRefLayerFrmWidth,  cRP.m_iFrameWidth  ) &&
        cRP.m_iScaledRefFrmHeight == gMin( cRP.m_iRefLayerFrmHeight, cRP.m_iFrameHeight )   )
      {
        resampling = false;
      }
      else
      {
        resampling = true;
      }

      //===== resample input frame =====
      if( ! bFinished )
      {
        resampleFrame( cFrame, cDownConvert, cRP, resamplingMethod, resamplingMode, resampling, upsampling, inputFrame != 0 );
      }
    }

    //===== write output frame =====
    if( ! bFinished )
    {
      Bool bWriteTwoFrames = ( numOutputFrames == 2 && ( maxNumOutputFrames == 0 || writtenFrames + 1 < maxNumOutputFrames ) );
      writeFrame( cFrame, outputFile, cRP.m_iFrameWidth, cRP.m_iFrameHeight, bWriteTwoFrames );
      writtenFrames += ( bWriteTwoFrames ? 2 : 1 );
      bFinished      = ( maxNumOutputFrames != 0 && writtenFrames == maxNumOutputFrames );
      fprintf( stderr, "\r%6d frames converted", writtenFrames );
    }
  }
  long  endTime           = clock();

  deleteFrame( cFrame );
  fclose     ( inputFile );
  fclose     ( outputFile );
  if( croppingParametersFile )
  {
    fclose   ( croppingParametersFile );
  }

  fprintf( stderr, "\n" );
  double deltaInSecond = (double)( endTime - startTime) / (double)CLOCKS_PER_SEC;
  fprintf( stderr, "in %.2lf seconds => %.0lf ms/frame\n", deltaInSecond, deltaInSecond / (double)writtenFrames * 1000.0 );
  if( writtenFrames < maxNumOutputFrames )
  {
    fprintf( stderr, "\nNOTE: less output frames generated than specified!!!\n\n" );
  }
  return 0;
}
