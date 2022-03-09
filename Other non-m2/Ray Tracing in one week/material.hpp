#ifndef MATERIAL
#define MATERIAL
#include "random.hpp"
#include "ray.hpp"
#include <algorithm>
class material {
	public:
		std::optional<std::pair<vec3, ray>> virtual scatter(const ray& r, const hit_record& rec) const = 0;
		virtual ~material() = default;
};

class lambertian: public material 
{
	public:
		std::optional<std::pair<vec3, ray>> scatter(const ray& r, const hit_record& rec) const override 
	{
		vec3 target = rec.p + rec.n + rd_unit_sphere();
		ray scattered { rec.p, target - rec.p };
		return  {std::pair{ albedo, scattered }};
	}
	lambertian(const vec3& a): albedo{a} {}
	vec3 albedo;
};

vec3 reflect (const vec3 &v, const vec3 &n)
{
	return v - (2*dot(v,n)*n);
}

class metal: public material
{
	public:
		std::optional<std::pair<vec3, ray>> scatter(const ray &r, const hit_record &rec) const override 
		{
			vec3 reflected = reflect(unit(r.d), rec.n);
		//	std::cout << rec.n.s() << '\n';
		
			ray scattered {rec.p, reflected + fuzz * rd_unit_sphere() };

			if (dot(scattered.d, rec.n) >  0) 
				return { std::pair{albedo, scattered} };
			return {};
		}

	metal(const vec3& a): albedo{a} {}
	metal(const vec3& a, float f): albedo{a}, fuzz{std::min(f,(float) 1.0)} {}
	vec3 albedo;
	float fuzz = 1.0;
};

std::optional<vec3> refract (const vec3 &v, const vec3& n, float ratio)
{
	vec3 uv = unit(v);
	float dt = dot(uv, n);
	float dis = 1.0 - ratio*ratio*(1 -dt*dt);
	if (dis > 0)
		return ratio*(uv - n*dt) - n*sqrt(dis);
	else
		return {};
}

float schlick(float cosine, float ref_idx) {
	float r0 = (1-ref_idx) / (1+ref_idx);
	r0 = r0*r0;
	return r0 + (1 -r0) * pow( (1-cosine), 5 );
}

class dielectric: public material
{
	public:
		std::optional<std::pair<vec3, ray>> scatter( const ray& r, const hit_record& rec) const override 
		{
			
			bool insider = dot( r.d, rec.n) > 0;
			float ni_over = insider ? this->ref_idx : 1.0/this->ref_idx;
			float direction = insider ? -1.0 : 1.0;
			float cosine = insider ? this->ref_idx * dot( r.d, rec.n ) / norm(r.d) : -1.0 * dot (r.d, rec.n ) / norm(r.d);
			vec3 attenuation {1.0, 1.0, 1.0 };
				
			if ( auto opt = refract ( r.d, direction * rec.n, ni_over); 
					opt && (float) rnd() >= schlick(cosine, ref_idx) )
				return { std::pair {attenuation, ray {rec.p, opt.value()}} };
			
			return { std::pair { attenuation, ray {rec.p, reflect( r.d, rec.n ) } } };
			
		}

	dielectric(float x): ref_idx {x} {}
	float ref_idx;
};



#endif
