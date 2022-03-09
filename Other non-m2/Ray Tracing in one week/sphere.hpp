#ifndef SPHERE
#define SPHERE
#include <limits>
#include <optional>
#include "hittable.hpp"
#include "material.hpp"
#include <memory>

class sphere: public hittable{
	public:
	using hittable::hit;
	vec3 c;
	float r;
	std::unique_ptr<material> mat;
	std::optional<hit_record> hit(const ray& r, std::pair<float, float> it) const override;
	sphere(const vec3& c, float r, std::unique_ptr<material> mat): c{c}, r{r}, mat{std::move(mat)} {}
};


std::optional<hit_record> sphere::hit(const ray& r, std::pair<float, float> it) const 
{
	vec3 oc = r.o -c;
	float a = dot(r.d,r.d);
	float b = 2*dot(oc,r.d);
	float c = dot(oc,oc) - this->r*this->r;
	float d = b*b - 4*a*c;
	if ( d > 0)
	{
		float r1 = (-b - sqrt(d))/(2.0*a);
		if( r1 > std::get<0>(it) && r1 < std::get<1>(it) )
				return hit_record {r1, r.p(r1), (r.p(r1) - this->c) / this->r, mat.get()};
		float r2 = (-b + sqrt(d))/(2.0*a);
		if( r2 > std::get<0>(it) && r2 < std::get<1>(it) )
				return hit_record {r2, r.p(r2), (r.p(r2)- this->c) / this->r, mat.get()};
	}
	return {};
}
#endif
