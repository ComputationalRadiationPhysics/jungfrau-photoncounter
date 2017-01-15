#include "Filecache.hpp"
#include "Upload.hpp"
#include <iostream>
#include <iomanip>

int main()
{ 
	Filecache fc(10280484 + 10000);
	//TODO: load pedestal init files and calibrate pedestal maps
	//std::vector<Datamap> fc.loadMaps("data_pool/px_101016/allpede_250us_1243__B_000000.dat", 1024, 512);
	std::vector<Pedestalmap> pedestal = fc.loadMaps("data_pool/px_101016/pedeMaps.bin", 1024, 512);
	std::vector<Datamap> data = fc.loadMaps("data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat", 1024, 512);
	std::vector<Gainmap> gain = fc.loadMaps("data_pool/px_101016/gainMaps_M022.bin", 1024, 512);

	Upload up(pedestal, gain, 1024, 512);

	up.upload(data);
	up.upload(data);
	up.upload(data);
	up.upload(data);
	up.upload(data);

	return 0;
}
