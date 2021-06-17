#ifndef INTERNAL_FUNCTION_PROGRAMMING_H_
#define INTERNAL_FUNCTION_PROGRAMMING_H_
#include <vector>
#include <algorithm>
template<class Container, class Functor>
auto operator|(Container&& container, Functor&& functor) {
    std::vector<typename Container::type> res;

    for(const auto& item : container) {
        res.push_back(functor(item));
    }
    return res;
}
#endif  // INTERNAL_FUNCTION_PROGRAMMING_H_