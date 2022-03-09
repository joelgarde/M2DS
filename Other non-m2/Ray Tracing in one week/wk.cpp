#include <iostream>
#include "ray.hpp"
#include "sphere.hpp"
#include "camera.hpp"
#include <vector>
#include "hittable_vec.hpp"
#include <memory>
#include "random.hpp"
#include <cmath>
inline vec3 blend(const vec3& a, const vec3& b, float t)
{
	return t*a+(1-t)*b;
}

inline vec3 color(const ray& r, const  hittable_vec &w, int depth = 0)
{	const vec3 c1 = {1, 1,1}; const vec3 c2 = {0.3, 0.5, 0.7};
	const float eps = 0.001;

	if( auto rec = w.hit(r, {eps, std::numeric_limits<float>::infinity()} ) )
	{
		if (auto next = rec->mat->scatter(r, rec.value()); next && depth < 50 )
		{
			auto [attenuation, scattered] = next.value();
			return attenuation * color( scattered, w, depth +1);
		} else
		{
			return vec3{1.0, 0.0, 0.0};
		}
	}

	float t = (r.d.y+1)/2; //scale back to 0,1	
	return blend(c1, c2,t);
}

int main()
{
	const int nx = 400;
	const int ny = 200;
	const int ns = 30;
//	
//	auto p1 = std::make_unique<sphere>( vec3 {0, 0, 1}, 0.5);
	auto m1 = std::make_unique<lambertian>( vec3 {0.3,0.3,0.8} );
	auto m2 = std::make_unique<lambertian>( vec3 {0.8,0.8,0.0} );
	auto m3 = std::make_unique<metal> ( vec3 {0.8,0.6,0.2}, 0.25 );
	auto m4 = std::make_unique<dielectric> ( 1.5 );
	auto m5 = std::make_unique<dielectric> ( 1.5 );
	auto p5 = std::make_unique<sphere> (vec3{-1, 0, 1},  -0.45, std::move(m5));
	auto p1 = std::make_unique<sphere>( vec3{0, -100.5, 1}, 100, std::move(m2));
	auto p2 = std::make_unique<sphere>( vec3{0, 0, 1}, 0.5, std::move(m1));
	auto p3 = std::make_unique<sphere>( vec3{1, 0, 1}, 0.5, std::move(m3));
	auto p4 = std::make_unique<sphere>( vec3{-1, 0, 1}, 0.5, std::move(m4));
	std::vector<std::unique_ptr<hittable>> pp {};
	pp.push_back(std::move(p1));
	pp.push_back(std::move(p2));
	pp.push_back(std::move(p3));
	pp.push_back(std::move(p4));
	pp.push_back(std::move(p5));
	hittable_vec hit_w {std::move(pp)};
	
	const camera cam {
		vec3 {0, 0, 1}, vec3 {-2, 2, -1}, vec3 {0, 1, 0},
		90.0, (float)nx/(float)ny};
	
	std::cout << "P3\n" << nx << " " << ny << "\n255\n";
	for(int j = 0; j < ny; ++j) {
		for(int i = 0; i < nx; ++i) {
			vec3 c {0,0,0};
			for(int s=0; s < ns; ++s)
			{
				float u = float(i+rnd()) / float(nx);
				float v = float(j+rnd()) / float(ny);
				ray r = cam.get_ray(u,v);
				c = c+color(r, hit_w);
			}
			c = c / float(ns);
			c = { sqrt(c.x), sqrt(c.y), sqrt(c.z)}; //gamma correction.
			c = c * 255.99;
			std::cout << int(c.x) << " " << int(c.y) << " " << int(c.z) << "\n";
		}
	}
	return 0;
}
