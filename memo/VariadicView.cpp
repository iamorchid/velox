#include <vector>

using namespace std;

// velox/expression/UdfTypeResolver.h
namespace facebook::velox::exec {

template <bool nullable, typename T>
class VariadicView;

template <typename T>
using NullFreeVariadicView = VariadicView<false, T>;

template <typename T>
using NullableVariadicView = VariadicView<true, T>;

namespace detail {
template <typename T>
struct resolver {
  using in_type = typename CppToType<T>::NativeType;
  using null_free_in_type = in_type;
  using out_type = typename CppToType<T>::NativeType;
};

template <typename T>
struct resolver<Variadic<T>> {
  using in_type = NullableVariadicView<T>;
  using null_free_in_type = NullFreeVariadicView<T>;
  // Variadic cannot be used as an out_type
};
}

}

// velox/type/Type.h
namespace facebook::velox {
template <typename UNDERLYING_TYPE>
struct Variadic {
  using underlying_type = UNDERLYING_TYPE;

  Variadic() = delete;
};
}

// velox/expression/VariadicView.h
namespace facebook::velox::exec {

template <typename T>
struct VectorReader;

template <bool returnsOptionalValues, typename T>
class VariadicView {
  using reader_t = VectorReader<T>;
  using element_t = typename std::conditional<
      returnsOptionalValues,
      typename reader_t::exec_in_t,
      typename reader_t::exec_null_free_in_t>::type;

 public:
  VariadicView(
      const std::vector<std::unique_ptr<reader_t>>* readers,
      vector_size_t offset)
      : readers_(readers), offset_(offset) {}

  using Element = typename std::
      conditional<returnsOptionalValues, OptionalAccessor<T>, element_t>::type;

  class ElementAccessor {
   public:
    using element_t = Element;
    using index_t = int;

    ElementAccessor(
        const std::vector<std::unique_ptr<reader_t>>* readers,
        vector_size_t offset)
        : readers_(readers), offset_(offset) {}

    Element operator()(vector_size_t index) const {
      if constexpr (returnsOptionalValues) {
        return Element{(*readers_)[index].get(), offset_};
      } else {
        return (*readers_)[index]->readNullFree(offset_);
      }
    }

   private:
    const std::vector<std::unique_ptr<reader_t>>* readers_;
    // This is only not const to support the assignment operator.
    vector_size_t offset_;
  };

  using Iterator = IndexBasedIterator<ElementAccessor>;

  Iterator begin() const {
    return Iterator{
        0, 0, (int)readers_->size(), ElementAccessor(readers_, offset_)};
  }

  Iterator end() const {
    return Iterator{
        (int)readers_->size(),
        0,
        (int)readers_->size(),
        ElementAccessor(readers_, offset_)};
  }
};

}