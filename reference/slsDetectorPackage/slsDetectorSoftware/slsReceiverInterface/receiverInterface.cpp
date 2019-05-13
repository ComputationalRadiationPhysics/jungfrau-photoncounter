#include "receiverInterface.h"


#include  <sys/types.h>
#include  <sys/shm.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <bitset>
#include <cstdlib>
#include <iostream>



receiverInterface::receiverInterface(MySocketTCP *socket):dataSocket(socket){}



receiverInterface::~receiverInterface(){}



int receiverInterface::sendString(int fnum, char retval[], char arg[]){
	int ret = slsDetectorDefs::FAIL;
	char mess[MAX_STR_LENGTH];
	memset(mess, 0, MAX_STR_LENGTH);

	dataSocket->SendDataOnly(&fnum,sizeof(fnum));
	dataSocket->SendDataOnly(arg,MAX_STR_LENGTH);
	dataSocket->ReceiveDataOnly(&ret,sizeof(ret));
	if (ret==slsDetectorDefs::FAIL){
		dataSocket->ReceiveDataOnly(mess,sizeof(mess));
		cprintf(RED, "Receiver returned error: %s", mess);
	}
	if(strstr(mess,"Unrecognized Function")==NULL)
		dataSocket->ReceiveDataOnly(retval,MAX_STR_LENGTH);

	return ret;
}



int receiverInterface::sendUDPDetails(int fnum, char retval[], char arg[3][MAX_STR_LENGTH]){
	char args[3][MAX_STR_LENGTH];
	int ret = slsDetectorDefs::FAIL;
	char mess[MAX_STR_LENGTH];
	memset(mess, 0, MAX_STR_LENGTH);

	dataSocket->SendDataOnly(&fnum,sizeof(fnum));
	dataSocket->SendDataOnly(arg,sizeof(args));
	dataSocket->ReceiveDataOnly(&ret,sizeof(ret));
	if (ret==slsDetectorDefs::FAIL){
		dataSocket->ReceiveDataOnly(mess,sizeof(mess));
	    cprintf(RED, "Receiver returned error: %s", mess);
	}
	else
		dataSocket->ReceiveDataOnly(retval,MAX_STR_LENGTH);

	return ret;
}


int receiverInterface::sendInt(int fnum, int &retval, int arg){
	int ret = slsDetectorDefs::FAIL;
	char mess[MAX_STR_LENGTH];
	memset(mess, 0, MAX_STR_LENGTH);

	dataSocket->SendDataOnly(&fnum,sizeof(fnum));
	dataSocket->SendDataOnly(&arg,sizeof(arg));
	dataSocket->ReceiveDataOnly(&ret,sizeof(ret));
	if (ret==slsDetectorDefs::FAIL){
		dataSocket->ReceiveDataOnly(mess,sizeof(mess));
	    cprintf(RED, "Receiver returned error: %s", mess);
	}
	if(strstr(mess,"Unrecognized Function")==NULL)
		dataSocket->ReceiveDataOnly(&retval,sizeof(retval));

	return ret;
}



int receiverInterface::getInt(int fnum, int &retval){
	int ret = slsDetectorDefs::FAIL;
	char mess[MAX_STR_LENGTH];
	memset(mess, 0, MAX_STR_LENGTH);

	dataSocket->SendDataOnly(&fnum,sizeof(fnum));
	dataSocket->ReceiveDataOnly(&ret,sizeof(ret));
	if (ret==slsDetectorDefs::FAIL){
		dataSocket->ReceiveDataOnly(mess,sizeof(mess));
	    cprintf(RED, "Receiver returned error: %s", mess);
	}
	if(strstr(mess,"Unrecognized Function")==NULL)
		dataSocket->ReceiveDataOnly(&retval,sizeof(retval));

	return ret;
}



int receiverInterface::sendInt(int fnum, int64_t &retval, int64_t arg){
	int ret = slsDetectorDefs::FAIL;
	char mess[MAX_STR_LENGTH];
	memset(mess, 0, MAX_STR_LENGTH);

	dataSocket->SendDataOnly(&fnum,sizeof(fnum));
	dataSocket->SendDataOnly(&arg,sizeof(arg));
	dataSocket->ReceiveDataOnly(&ret,sizeof(ret));
	if (ret==slsDetectorDefs::FAIL){
		dataSocket->ReceiveDataOnly(mess,sizeof(mess));
	    cprintf(RED, "Receiver returned error: %s", mess);
	}
	if(strstr(mess,"Unrecognized Function")==NULL)
		dataSocket->ReceiveDataOnly(&retval,sizeof(retval));

	return ret;
}



int receiverInterface::sendIntArray(int fnum, int64_t &retval, int64_t arg[2], char mess[]){
	int64_t args[2];
	int ret = slsDetectorDefs::FAIL;
	memset(mess, 0, MAX_STR_LENGTH);

	dataSocket->SendDataOnly(&fnum,sizeof(fnum));
	dataSocket->SendDataOnly(arg,sizeof(args));
	dataSocket->ReceiveDataOnly(&ret,sizeof(ret));
	if (ret==slsDetectorDefs::FAIL){
		dataSocket->ReceiveDataOnly(mess,MAX_STR_LENGTH);
	    cprintf(RED, "Receiver returned error: %s", mess);
	}
	if(strstr(mess,"Unrecognized Function")==NULL)
		dataSocket->ReceiveDataOnly(&retval,sizeof(retval));

	return ret;
}



int receiverInterface::sendIntArray(int fnum, int &retval, int arg[2]){
	int args[2];
	int ret = slsDetectorDefs::FAIL;
	char mess[MAX_STR_LENGTH];
	memset(mess, 0, MAX_STR_LENGTH);

	dataSocket->SendDataOnly(&fnum,sizeof(fnum));
	dataSocket->SendDataOnly(arg,sizeof(args));
	dataSocket->ReceiveDataOnly(&ret,sizeof(ret));
	if (ret==slsDetectorDefs::FAIL){
		dataSocket->ReceiveDataOnly(mess,sizeof(mess));
	    cprintf(RED, "Receiver returned error: %s", mess);
	}
	if(strstr(mess,"Unrecognized Function")==NULL)
		dataSocket->ReceiveDataOnly(&retval,sizeof(retval));

	return ret;
}



int receiverInterface::getInt(int fnum, int64_t &retval){
	int ret = slsDetectorDefs::FAIL;

	dataSocket->SendDataOnly(&fnum,sizeof(fnum));
	dataSocket->ReceiveDataOnly(&ret,sizeof(ret));
	dataSocket->ReceiveDataOnly(&retval,sizeof(retval));

	return ret;
}


int receiverInterface::getLastClientIP(int fnum, char retval[]){
	int ret = slsDetectorDefs::FAIL;

	dataSocket->SendDataOnly(&fnum,sizeof(fnum));
	dataSocket->ReceiveDataOnly(&ret,sizeof(ret));
	dataSocket->ReceiveDataOnly(retval,INET_ADDRSTRLEN);

	return ret;
}



int receiverInterface::executeFunction(int fnum,char mess[]){
	int ret = slsDetectorDefs::FAIL;
	memset(mess, 0, MAX_STR_LENGTH);

	dataSocket->SendDataOnly(&fnum,sizeof(fnum));
	dataSocket->ReceiveDataOnly(&ret,sizeof(ret));
	if (ret==slsDetectorDefs::FAIL){
		dataSocket->ReceiveDataOnly(mess,MAX_STR_LENGTH);
	    cprintf(RED, "Receiver returned error: %s", mess);
	}

	return ret;
}



int receiverInterface::sendROI(int fnum, int n, slsReceiverDefs::ROI roiLimits[]) {
	int ret = slsDetectorDefs::FAIL;
	char mess[MAX_STR_LENGTH];
	memset(mess, 0, MAX_STR_LENGTH);

	dataSocket->SendDataOnly(&fnum,sizeof(fnum));
	dataSocket->SendDataOnly(&n,sizeof(n));
	dataSocket->SendDataOnly(roiLimits,n * sizeof(slsReceiverDefs::ROI));
	dataSocket->ReceiveDataOnly(&ret,sizeof(ret));
	if (ret==slsDetectorDefs::FAIL){
		dataSocket->ReceiveDataOnly(mess,sizeof(mess));
	    cprintf(RED, "Receiver returned error: %s", mess);
	}
	return ret;
}


