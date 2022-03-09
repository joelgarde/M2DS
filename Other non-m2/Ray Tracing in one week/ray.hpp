#ifndef RAY
#define RAY
#include "vec3.cpp"

struct ray
{
	vec3 o;
	vec3 d;
	vec3 p(float t) const {return o + t*d;}
};
#endif
