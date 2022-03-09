#ifndef HITTABLELIST
#define HITTABLELIST

#include "hittable.hpp"
#include "ray.hpp"
#include <optional>
#include <memory>

class hittable_vec: hittable {
	public:
	using hittable::hit;
	hittable_vec(std::vector<std::unique_ptr<hittable>>&& u ): v{std::move(u)} {}	
//	hittable_vec(std::initializer_list<std::unique_ptr<hittable>> &&u ): v{std::move(u)} {} can't really move initializer list of uniq. ptr.
	std::vector<std::unique_ptr<hittable>> v;
	std::optional<hit_record> hit(const ray& r, std::pair<float, float> it) const override;
};

std::optional<hit_record> hittable_vec::hit(const ray& r, std::pair<float, float> it) const
{
	std::optional<hit_record> result = std::nullopt;
	auto min_so_far = std::get<1>(it);
	for (const auto& h_ptr: v)
		if( auto record = h_ptr->hit(r, {std::get<0>(it), min_so_far} ))
			result = record, min_so_far = result->t;
	return result;

}

#endif
