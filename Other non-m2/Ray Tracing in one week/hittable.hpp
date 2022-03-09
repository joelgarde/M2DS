#ifndef HITTABLE
#define HITTABLE
#include <optional>
#include <limits>
#include <memory>

class material;

struct hit_record 
{
	const material *mat;
	float t;
	vec3 p;
	vec3 n;
	hit_record(float t, const vec3& p, const vec3& n, const material* mat): mat(mat), t{t}, p{p}, n{n}{}
};


class hittable{
	public:
	virtual std::optional<hit_record> hit(const ray& r, std::pair<float, float> it = 
			{ -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity()}) const = 0;
	virtual ~hittable() = default;	
	std::optional<hit_record> hit(const ray& r) const { 
		return this->hit(r, 
			{ 0, +std::numeric_limits<float>::infinity() });
	}
			
};

#endif
