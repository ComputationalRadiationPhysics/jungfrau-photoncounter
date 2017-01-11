#include "Gainmap.h"
#include "Pedestalmap.h"
#include <cassert>
#include <iostream>

int main()
{
	/* just some redundant tests... */
    Gainmap g(100, 100);
	Pedestalmap p(10, 10);
	p(0, 0) = 0;
	p(1, 0) = 1;
	p(2, 0) = 2;
	std::cout << p.data()[0] << " " << p.data()[1] << " " << p.data()[2];
	return 0;
}
