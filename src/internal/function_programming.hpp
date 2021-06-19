#ifndef INTERNAL_FUNCTION_PROGRAMMING_H_
#define INTERNAL_FUNCTION_PROGRAMMING_H_
#include <vector>
#include <algorithm>
template<class Container, class Functor>
auto operator|(Container&& container, Functor&& functor) {
    using Container_Type = std::remove_reference_t<Container>;
    std::vector<typename std::result_of_t<Functor(typename Container_Type::value_type)>> res;

    for(const auto& item : container) {
        res.push_back(functor(item));
    }
    return res;
}
#endif  // INTERNAL_FUNCTION_PROGRAMMING_H_