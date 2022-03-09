#ifndef CAMERA
#define CAMERA

#include <cmath>
#include "ray.hpp"

class camera
{
	vec3 o_ {0,0,0}; 
	vec3 c_ {-1, 1, 1};
	vec3 h_ {2,0,0};
	vec3 v_ {0,-2,0};
	public:
	camera (vec3 look_at, vec3 look_from, vec3 vup, float vfov, float aspect) {
		float theta = vfov * 3.14159265/180;
		float half_height = (float) tan( theta / 2 );
		float half_width = aspect * half_height;
		
		//define the new basis.
		o_ = look_from;		
		vec3 w = unit( look_at - look_from );
		vec3 u = unit(cross( vup, w ));
		vec3 v = unit(cross( w, u ));
		c_ = o_ + w - half_width*u  +half_height*v;
		h_ = 2*half_width*u;
		v_ = -2*half_height*v;
	//	std::cout << o_ <<" "<< c_ <<" "<< h_<<" "<< v_<< '\n';
	}
	inline ray get_ray(float u, float v) const {
		return ray {o_, c_ + u * h_ + v * v_ -o_};
	}
};

#endif
