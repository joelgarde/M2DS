#ifndef RANDOM
#define RANDOM

#include<random>

inline double rnd();

vec3 rd_unit_sphere()
{
	vec3 u = (2 * vec3 {(float) rnd(), (float) rnd(), (float) rnd()}) - 1;
//	std::cout << u.s() << '\n';
	if ( u.s() <= 1 )
		return u;
	else
		return rd_unit_sphere();
}

inline double rnd() 
{
	static std::random_device rd{};
	static std::ranlux24_base gen(rd());
	static std::uniform_real_distribution<double> d {0.0, 1.0};
	return d(gen);
}
#endif
