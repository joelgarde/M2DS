#include "random.hpp"
#include <iostream>

int main()
{
	for(int i = 0; i < 10; ++i)
		std::cout << rnd() << ' ';
	std::cout << "\n";
}

