#include <math.h>
#include <iostream>
struct vec3 {
	float x;
	float y; 
	float z;
	float s() const;
};

vec3 operator-(const vec3& a)
{
	return {-a.x, -a.y, -a.z};
}
vec3 operator+(const vec3& a, const vec3& b)
{
	return {a.x+b.x, a.y+b.y, a.z+b.z};
}
vec3 operator-(const vec3& a, const vec3& b)
{
	return a+(-b);
}
vec3 operator*(const vec3& a, float t)
{
	return {a.x*t, a.y*t, a.z*t};
}
vec3 operator*(float t, const vec3& a)
{
	return a*t;
}
vec3 operator*(const vec3& a, const vec3& b)
{
	return {a.x*b.y, a.y*b.y,a.z*b.z};
}
vec3 operator+(const vec3& a, float t)
{
	return {a.x+t, a.y+t, a.z+t};
}
vec3 operator+(float t, const vec3& a) {return a+t;}
vec3 operator-(const vec3& a, float t)
{
	return {a.x-t, a.y-t, a.z-t};
}
vec3 operator-(float t, const vec3& a)
{
	return a-t;
}
vec3 operator/(const vec3& a, float t){ return a * (1/t);}

float dot(const vec3& a, const vec3& b)
{
	return a.x*b.x+a.y*b.y+a.z*b.z;
}
vec3 operator%(const vec3& a, const vec3& b)
{
	return  {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; 
}

float vec3::s() const
{
	return x*x+y*y+z*z;
}

float norm(const vec3& x)
{
	return sqrt(x.s());
}

vec3 unit(const vec3& x)
{
	return x/norm(x);
};

bool operator<(const vec3& a, const vec3& b)
{
	return (a.s() < b.s());
}
bool operator>(const vec3& a, const vec3& b)
{
	return b < a;
}
std::ostream& operator<<(std::ostream &os, const vec3& a)
{
	os << a.x << " " << a.y << " " << a.z;
	return os;
}

vec3 cross(const vec3& a, const vec3& b) {
	return a % b;
}
std::istream& operator>>(std::istream &is, vec3& a)
{
	is >> a.x >> a.y >> a.z;
	return is;
}
