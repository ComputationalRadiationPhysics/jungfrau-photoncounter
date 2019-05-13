

INCDIR=-I. -I../ -I../interpolations  -I../interpolations/etaVEL  -I../dataStructures -I../../slsReceiverSoftware/include 

LDFLAG= ../tiffIO.cpp -L/trinity/shared/pkg/tools/libtiff/4.0.9/lib -L/usr/lib64/ -lpthread -lm  -lstdc++ -L.  -pthread -lrt   -ltiff 
#CFLAGS=$(CFLAGS) -O3 -march=native
CFLAGS= -g -ggdb -O2 #-fsanitize=address
CPP=g++

all: moenchClusterFinderScrumbled  moenchClusterFinderReordered moenchReorderImage


moenchReorderImage :  moench03ReorderImage.cpp   $(INCS) clean
		   $(CPP) -o moenchReorderImage   moench03ReorderImage.cpp $(CFLAGS) $(LDFLAG) $(INCDIR) -DSAVE_ALL  -DNEWRECEIVER -D VERSION_V1



moenchClusterFinderScrumbled :  moench03ClusterFinder.cpp  $(INCS) clean
			 $(CPP) -o moenchClusterFinderScrubled   moench03ClusterFinder.cpp $(CFLAGS) $(LDFLAG) $(INCDIR) -DSAVE_ALL  -DNEWRECEIVER -D VERSION_V1


moenchClusterFinderReordered :  moench03ClusterFinder.cpp  $(INCS) clean
			 $(CPP) -o moenchClusterFinderReordered   moench03ClusterFinder.cpp $(CFLAGS) $(LDFLAG) $(INCDIR) -DSAVE_ALL  -DREORDERED 

clean: 	
	rm -f  moenchClusterFinderScrubled moenchClusterFinderReordered moenchReorderImage




