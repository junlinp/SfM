#ifndef INTERNAL_FUNCTION_PROGRAMMING_H_
#define INTERNAL_FUNCTION_PROGRAMMING_H_
#include <algorithm>
#include <vector>

//
//
// std::vector<a> container1
// functor : a -> b
//
// filter_functor : b -> bool
// container1 | Transform(functor) | Filter(filter_functor) | ToVector();
//
//
// So ContainerOutput operator|(Container1, TransformBuilder)
// So ContainerOutput operator|(Container1, Output)
//
//
template <class InputIterator, class Functor, class OutputType>
struct TransformContainerIterator
    : std::iterator<std::forward_iterator_tag, OutputType> {
  TransformContainerIterator(InputIterator iterator, Functor functor)
      : iterator_(iterator), functor_(functor){};

  TransformContainerIterator operator++() {
    TransformContainerIterator temp(iterator_, functor_);
    ++iterator_;
    return temp;
  }

  bool operator!=(const TransformContainerIterator& rhs) const {
    return iterator_ != rhs.iterator_;
  }

  OutputType operator*() const { return functor_(*iterator_); }

  InputIterator iterator_;
  Functor functor_;
};

template <class InputContainer, class Functor>
class TransformContainer {
 public:
  TransformContainer(InputContainer container, Functor functor)
      : container_(std::forward<InputContainer>(container)),
        functor_(std::forward<Functor>(functor)) {}
  using InputType = typename std::decay_t<InputContainer>::value_type;
  using OutputType = std::result_of_t<Functor(InputType)>;
  using value_type = OutputType;
  using iterator = TransformContainerIterator<typename std::decay_t<InputContainer>::iterator,
                                      Functor, OutputType>;

  iterator begin() { return iterator(container_.begin(), functor_); };
  iterator end() { return iterator(container_.begin(), functor_); }

 private:
  InputContainer container_;
  Functor functor_;
};

template <class Functor>
class TransformContainerBuilder {
 public:
  TransformContainerBuilder(Functor functor)
      : functor_(std::forward<Functor>(functor)) {}

  template <class InputContainer>
  auto Build(InputContainer&& container) {
    return TransformContainer<InputContainer, Functor>(
        std::forward<InputContainer>(container),
        std::forward<Functor>(functor_));
  }
  Functor functor_;
};

template<class Functor>
auto Transform(Functor&& functor) {
    return TransformContainerBuilder<Functor>(std::forward<Functor>(functor));
}

template <class Container, class Functor>
auto operator|(Container&& container, TransformContainerBuilder<Functor>&& builder) {
    return builder.Build(container);
}

struct ToVectorBuilder{};

ToVectorBuilder ToVector() {
    return ToVectorBuilder();
}
template <class Container>
auto operator|(Container&& container, ToVectorBuilder&& /**/) {
    using Type = typename std::decay_t<Container>::value_type;
    std::vector<Type> res;
    for(auto item : container) {
        res.emplace_back(item);
    }
    return res;
}

#endif  // INTERNAL_FUNCTION_PROGRAMMING_H_