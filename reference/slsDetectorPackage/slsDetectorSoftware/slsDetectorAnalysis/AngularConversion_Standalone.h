
#ifndef ANGULARCONVERSIONSTD_H
#define ANGULARCONVERSIONSTD_H


//#include "slsDetectorBase.h"
#include "sls_detector_defs.h"
#include <string>
#include <fstream>



  //double angle(int ichan, double encoder, double totalOffset, double conv_r, double center, double offset, double tilt, int direction)


using namespace std;

/**
   @short Angular conversion constants needed  for a detector module
 */
typedef struct  {
  double center;  /**< center of the module (channel at which the radius is perpendicular to the module surface) */
  double ecenter; /**< error in the center determination */
  double r_conversion;  /**<  detector pixel size (or strip pitch) divided by the diffractometer radius */
  double er_conversion;  /**< error in the r_conversion determination */
  double offset; /**< the module offset i.e. the position of channel 0 with respect to the diffractometer 0 */
  double eoffset; /**< error in the offset determination */
  double tilt; /**< ossible tilt in the orthogonal direction (unused)*/
  double etilt; /**< error in the tilt determination */
} angleConversionConstant;


/**

@short methods to set/unset the angular conversion and merge the data
class containing the methods to set/unset the angular conversion and merge the data


The angular conversion itself is defined by the angle() function defined in usersFunctions.cpp
   
*/
class angularConversion : public virtual slsDetectorDefs { //: public virtual slsDetectorBase {

 public:
  /** default constructor */
  angularConversion(int*, double*, double*, double*, double*);
  /** virtual destructor */
  virtual ~angularConversion();
 


  //virtual int readAngularConversion(string fname)=0;




  /**
   
      reads an angular conversion file
      \param fname file to be read
      \param nmod number of modules (maximum) to be read
      \param angOff pointer to array of angleConversionConstants
      \returns OK or FAIL
  */
   static int readAngularConversion(string fname, int nmod, angleConversionConstant *angOff); 
   
   /**
      reads an angular conversion file
      \param ifstream input file stream to be read
      \param nmod number of modules (maximum) to be read
      \param angOff pointer to array of angleConversionConstants
      \returns OK or FAIL
      
  */
   static int readAngularConversion(ifstream& ifs, int nmod, angleConversionConstant *angOff);
  /**
     writes an angular conversion file
      \param fname file to be written
      \param nmod number of modules to be written
      \param angOff pointer to array of angleConversionConstants
      \returns OK or FAIL
  */
   static int writeAngularConversion(string fname, int nmod, angleConversionConstant *angOff);

 /**
     writes an angular conversion file
      \param ofstream output file stream
      \param nmod number of modules to be written
      \param angOff pointer to array of angleConversionConstants
      \returns OK or FAIL
  */
   static int writeAngularConversion(ofstream& ofs, int nmod, angleConversionConstant *angOff);

   /** 
       sets the arrays of the merged data to 0. NB The array should be created with size nbins >= 360./getBinSize(); 
       \param mp already merged postions
       \param mv already merged data
       \param me already merged errors (squared sum)
       \param mm multiplicity of merged arrays
       \param nbins number of bins
       \returns OK or FAIL
   */
   static int resetMerging(double *mp, double *mv,double *me, int *mm, int nbins);
   /** 
       sets the arrays of the merged data to 0. NB The array should be created with size >= 360./getBinSize(); 
       \param mp already merged postions
       \param mv already merged data
       \param me already merged errors (squared sum)
       \param mm multiplicity of merged arrays
       \returns OK or FAIL
   */
   int resetMerging(double *mp, double *mv,double *me, int *mm);
  
   /** 
       creates the arrays for merging the data and sets them to 0. 
   */
   int resetMerging();

  /** 
      merge dataset
      \param p1 angular positions of dataset
      \param v1 data
      \param e1 errors
      \param mp already merged postions
      \param mv already merged data
      \param me already merged errors (squared sum)
      \param mm multiplicity of merged arrays
      \param nchans number of channels
      \param binsize size of angular bin
      \param nb number of angular bins
      \param badChanMask badchannelmask (if NULL does not correct for bad channels)
      \returns OK or FAIL
  */

   static int  addToMerging(double *p1, double *v1, double *e1, double *mp, double *mv,double *me, int *mm, int nchans, double binsize,int nb, int *badChanMask );
   
  /** 
      merge dataset
      \param p1 angular positions of dataset
      \param v1 data
      \param e1 errors
      \param mp already merged postions
      \param mv already merged data
      \param me already merged errors (squared sum)
      \param mm multiplicity of merged arrays
      \param badChanMask badchannelmask (if NULL does not correct for bad channels)
      \returns OK or FAIL
  */

   int  addToMerging(double *p1, double *v1, double *e1, double *mp, double *mv,double *me, int *mm, int *badChanMask);
  /** 
      merge dataset
      \param p1 angular positions of dataset
      \param v1 data
      \param e1 errors
      \param badChanMask badchannelmask (if NULL does not correct for bad channels)
      \returns OK or FAIL
  */

   int addToMerging(double *p1, double *v1, double *e1,int *badChanMask);

   /**
      calculates the "final" positions, data value and errors for the merged data
      \param mp already merged postions
      \param mv already merged data
      \param me already merged errors (squared sum)
      \param mm multiplicity of merged arrays
      \param nb number of bins
      \returns FAIL or the number of non empty bins (i.e. points belonging to the pattern)
  */

   static int finalizeMerging(double *mp, double *mv,double *me, int *mm, int nb);
   /**
      calculates the "final" positions, data value and errors for the merged data
      \param mp already merged postions
      \param mv already merged data
      \param me already merged errors (squared sum)
      \param mm multiplicity of merged arrays
      \returns FAIL or the number of non empty bins (i.e. points belonging to the pattern)
  */

   int finalizeMerging(double *mp, double *mv,double *me, int *mm);

/**
      calculates the "final" positions, data value and errors for the merged data
      \returns FAIL or the number of non empty bins (i.e. points belonging to the pattern)
  */

   int finalizeMerging();


  /**
      set detector global offset
      \param f global offset to be set
      \returns actual global offset
  */
  double setGlobalOffset(double f){return setAngularConversionParameter(GLOBAL_OFFSET,f);};

 
  /**
      set detector fine offset
      \param f global fine to be set
      \returns actual fine offset
  */
  double setFineOffset(double f){return setAngularConversionParameter(FINE_OFFSET,f);};
  
  /**
      get detector fine offset
      \returns actual fine offset
  */
  double getFineOffset(){return getAngularConversionParameter(FINE_OFFSET);};
  
  /**
      get detector global offset
      \returns actual global offset
  */
  double getGlobalOffset(){return getAngularConversionParameter(GLOBAL_OFFSET);};

  /** 
      
      set detector bin size
      \param bs bin size to be set
      \returns actual bin size
  */
  double setBinSize(double bs){if (bs>0) nBins=360/bs; return setAngularConversionParameter(BIN_SIZE,bs);};

  /** 
      get detector bin size
      \returns detector bin size used for merging (approx angular resolution)
  */
  double getBinSize() {return getAngularConversionParameter(BIN_SIZE);};



  /** 
      
      get angular direction
      \returns actual angular direction (1 is channel number increasing with angle, -1 decreasing)
  */
  int getAngularDirection(){return (int)getAngularConversionParameter(ANGULAR_DIRECTION);};

  
  /** 
      
      set angular direction
      \param d angular direction to be set (1 is channel number increasing with angle, -1 decreasing)
      \returns actual angular direction (1 is channel number increasing with angle, -1 decreasing)
  */
  int setAngularDirection(int d){return (int)setAngularConversionParameter(ANGULAR_DIRECTION, (double)d);};

  /**
     \returns  number of angular bins in the merging (360./binsize)
  */
  int getNumberOfAngularBins(){return nBins;};



  /**
     set angular conversion parameter
     \param c parameter type (globaloffset, fineoffset, binsize, angular direction, move flag)
     \param v value to be set
     \returns actual value
  */
   double setAngularConversionParameter(angleConversionParameter c, double v);
 /**
     get angular conversion parameter
     \param c parameter type (globaloffset, fineoffset, binsize, angular direction, move flag)
     \returns actual value
  */
   double getAngularConversionParameter(angleConversionParameter c);
   
 


  /** 
      set  positions for the acquisition
      \param nPos number of positions
      \param pos array with the encoder positions
      \returns number of positions
  */
  virtual int setPositions(int nPos, double *pos);
   /** 
      get  positions for the acquisition
      \param pos array which will contain the encoder positions
      \returns number of positions
  */
  virtual int getPositions(double *pos=NULL);
 
  /**
     deletes the array of merged data
     \returns OK
  */
  int deleteMerging();

  /**
     \returns pointer to the array o merged positions
  */
  double *getMergedPositions(){return mergingBins;};
  /**
     \returns pointer to the array of merged counts
  */
  double *getMergedCounts(){return mergingCounts;};
  /**
     \returns pointer to the array of merged errors
  */
  double *getMergedErrors(){return mergingErrors;};
  



/*   /\** */
/*      reads teh angular conversion file for the (multi)detector and writes it to shared memory */
/*   *\/ */
/*   virtual int readAngularConversionFile(string fname="")=0; */

/*   /\** */
/*      get angular conversion */
/*      \param direction reference to diffractometer direction */
/*       \param angconv array that will be filled with the angular conversion constants */
/*       \returns 0 if angular conversion disabled, >0 otherwise */
/*   *\/ */
  virtual int getAngularConversion(int &direction,  angleConversionConstant *angoff=NULL); 

/*    /\**  */
/*        pure virtual function */
/*        \param file name to be written (nmod and array of angular conversion constants default to the ones ot the slsDetector */
/*    *\/ */
/*    virtual int writeAngularConversion(string fname)=0; */




/*   /\** */
/*      \returns number of modules of the (multi)detector */
/*   *\/ */
/*   virtual int getNMods()=0; */

/*   /\** */
/*      returns number of channels in the module */
/*      \param imod module number */
/*      \returns number of channels in the module */
/*   *\/ */


  static int defaultGetNumberofChannel(int nch, void *p=NULL ) { ((angularConversion*)p)->setTotalNumberOfChannels(nch); return 0;};
  static int defaultGetChansPerMod(int imod=0, void *p=NULL){ ((angularConversion*)p)->setChansPerMod(0,imod); return 0;};

  int setChansPerMod(int nch, int imod=0){if (nch<0) return -1; if (imod>=0 && imod<MAXMODS*MAXDET) {chansPerMod[imod]=nch; return chansPerMod[imod];} else return -1;};
		
  int setTotalNumberOfChannels(int i){if (i>=0){ totalNumberOfChannels=i; return totalNumberOfChannels;} else return -1;};
  

/*   /\** */
/*      get the angular conversion  contant of one modules */
/*      \param imod module number */
/*      \returns pointer to the angular conversion constant */
/*   *\/ */
/*   virtual angleConversionConstant *getAngularConversionPointer(int imod=0)=0; */
/*     /\** */
/*      \param imod module number */
/*      \returns move flag of the module (1 encoder is added to the angle, 0 not) */
/*   *\/ */
/*   virtual int getMoveFlag(int imod)=0; */

/*   /\** */
/*      enables/disable the angular conversion */
/*      \param i 1 sets, 0 unsets,, -1 gets */
/*      \returns actual angular conversion flag */
/*   *\/ */
/*   virtual int setAngularCorrectionMask(int i=-1)=0; */
  


  /**
     converts channel number to angle
     \param pos encoder position
     \returns array of angles corresponding to the channels
  */
  double* convertAngles(double pos);
  /**
     converts channel number to angle for the current encoder position
     \returns array of angles corresponding to the channels
  */
  double *convertAngles(){return convertAngles(currentPosition);};
  

  /**
     returns number of positions
  */
  int getNumberOfPositions() {return *numberOfPositions;};

  int getChansPerMods(int imod) { return chansPerMod[imod];};
  int getTotalNumberofChannels(){ return totalNumberOfChannels;};


  void incrementPositionIndex() {currentPositionIndex++;};

  void registerCallBackGetChansPerMod(int (*func)(int, void *),void *arg){ getChansPerModule=func;pChpermod=arg;};
  void registerCallBackGetNumberofChannel(int (*func)(int, void *),void *arg){ getNumberofChannel=func;pNumberofChannel=arg;};

 protected:
 

	
 private:


  int (*getChansPerModule)(int, void*);
  int (*getNumberofChannel)(int, void*);
	
  void *pChpermod,*angPtr,*pNumberofChannel;


  /** merging bins */
  double *mergingBins;
  
  /** merging counts */
  double *mergingCounts;
  
  /** merging errors */
  double *mergingErrors;
  
  /** merging multiplicity */
  int *mergingMultiplicity;
  
  double (*angle)(double, double, double, double, double, double, double, int);
  


  int totalNumberOfChannels;

  int moveFlag[MAXMODS*MAXDET];

  /** pointer to number of positions for the acquisition*/
   int *numberOfPositions;
   
  /** pointer to the detector positions for the acquisition*/
   double *detPositions;
      
   
   /** pointer to angular bin size*/
   double *binSize;

  int *correctionMask;
  int chansPerMod[MAXMODS*MAXDET];
  int nMod;
  angleConversionConstant angConvs[MAXMODS*MAXDET];
  int directions[MAXMODS*MAXDET];
   

   /** pointer to beamline fine offset*/
   double *fineOffset;
   /** pointer to beamline global offset*/
   double *globalOffset;
   /** pointer to beamline angular direction*/
   int *angDirection;

   /** pointer to detector move flag (1 moves with encoder, 0 not)*/
   //   int *moveFlag;

   /** number of bins for angular conversion (360./binsize)*/
   int nBins;
   
   

  /**
     current position of the detector
  */
  double currentPosition;
  /**
     current position index of the detector
  */
  int currentPositionIndex;
  

  /**
     returns current position index
  */
  int getCurrentPositionIndex() {return currentPositionIndex;};

 
 


 void registerAngleFunctionCallback(double( *fun)(double, double, double, double, double, double, double, int),void* arg) {angle = fun; angPtr=arg;};
  
  
};

#endif
