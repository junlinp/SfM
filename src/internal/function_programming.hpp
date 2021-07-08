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

template<class InputIterator, class Predicate, class OutputType>
struct FilterIterator : std::iterator<std::forward_iterator_tag, OutputType> {
    FilterIterator(InputIterator iterator, Predicate predicate, const InputIterator& begin, const InputIterator& end) : iterator_(iterator), predicate_(predicate), begin_(begin), end_(end) {
        iterator_ = std::find_if(iterator_, end_, predicate_);
    }

    FilterIterator operator++() {
        FilterIterator temp(iterator_, predicate_, begin_, end_);
        ++iterator_;
        iterator_ = std::find_if(iterator_, end_, predicate_);
        return temp;
    }

    bool operator!=(const FilterIterator& rhs) const {
        return iterator_ != rhs.iterator;
    }

    OutputType operator*() const {
        return *iterator_;
    }

    InputIterator iterator_, begin_, end_;
    Predicate predicate_;
};

template<class Container, class Predicate>
class FilterContainer {
 public:
  FilterContainer(Container container, Predicate predicate)
      : container_(std::forward<Container>(container)),
        predicate_(std::forward<Predicate>(predicate)) {}

        using value_type = typename std::decay_t<Container>::value_type;
        using origin_iterator = typename std::decay_t<Container>::iterator;
        using iterator = FilterIterator<origin_iterator, Predicate, value_type>;

    auto begin() {
        return iterator(container_.begin(), predicate_, container_.begin(), container_.end());
    }

    auto end() {
        return iterator(container_.end(), predicate_, container_.begin(), container_.end());
    }

 private:
  Container container_;
  Predicate predicate_;
};
template<class Predicate>
class FilterBuilder {
public:
    FilterBuilder(Predicate predicate) : predicate_(std::forward<Predicate>(predicate)) {}

    template<class Container>
    auto Build(Container container) {
      return FilterContainer<Container, Predicate>(
          std::forward<Container>(container),
          std::forward<Predicate>(predicate_));
    }
private:
 Predicate predicate_;
};

template<class Predicate>
auto Filter(Predicate&& predicate) {
    return FilterBuilder<Predicate>(std::forward<Predicate>(predicate));
}

template<class Container, class Predicate>
auto operator|(Container&& container, FilterBuilder<Predicate>&& builder) {
    return builder.template Build<Container>(std::forward<Container>(container));
}

#endif  // INTERNAL_FUNCTION_PROGRAMMING_H_