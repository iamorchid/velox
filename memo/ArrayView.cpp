#include <vector>

using namespace std;

// 参考 ArrayViewTest.cpp中的 intArrayTest

template <typename T>
struct ArraySum {
  // 参见 velox/functions/Macros.h
  VELOX_DEFINE_FUNCTION_TYPES(T);

  bool call(const int64_t& output, const arg_type<Array<int64_t>>& array) {
    output = 0;
    for(const auto& element : array) {
      if (element.has_value()) {
        output += element.value();
      }
    }
    return true;
  }
};

// velox/functions/Registerer.h
namespace facebook::velox {

template <template <class> typename Func, typename TReturn, typename... TArgs>
void registerFunction(const std::vector<std::string>& aliases = {}) {
  using funcClass = Func<exec::VectorExec>;
  using holderClass = core::UDFHolder<
      funcClass,
      exec::VectorExec,
      TReturn,
      ConstantChecker<TArgs...>,
      typename UnwrapConstantType<TArgs>::type...>;
  exec::registerSimpleFunction<holderClass>(aliases);
}

}

// velox/expression/UdfTypeResover.h
namespace facebook::velox::exec {

template <bool nullable, typename V>
class ArrayView;

template <typename T>
using NullFreeArrayView = ArrayView<false, T>;

template <typename T>
using NullableArrayView = ArrayView<true, T>;

namespace detail {

template <typename T>
struct resolver {
  using in_type = typename CppToType<T>::NativeType;
  using null_free_in_type = in_type;
  using out_type = typename CppToType<T>::NativeType;
};

template <typename V>
struct resolver<Array<V>> {
  using in_type = NullableArrayView<V>;
  using null_free_in_type = NullFreeArrayView<V>;
  using out_type = ArrayWriter<V>;
};

}

struct VectorExec {
  template <typename T>
  using resolver = typename detail::template resolver<T>;
};

}


// velox/type/Type.h
namespace facebook::velox {

template <typename ELEMENT>
struct Array {
  using element_type = ELEMENT;

  static_assert(
      !isVariadicType<element_type>::value,
      "Array elements cannot be Variadic");

 private:
  Array() {}
};

}


// velox/expression/ComplexViewTypes.h
namespace facebook::velox::exec {

template <typename T>
class OptionalAccessor {
 public:
  using element_t = typename VectorReader<T>::exec_in_t;

  explicit operator bool() const {
    return has_value();
  }

  bool has_value() const {
    return reader_->isSet(index_);
  }

  element_t value() const {
    VELOX_DCHECK(has_value());
    return (*reader_)[index_];
  }
};

template <bool returnsOptionalValues, typename V>
class ArrayView {
 public:
  // 这个reader对应的是ArrayVector的base vector (即elements_)
  using reader_t = VectorReader<V>;

  using element_t = typename std::conditional<
      returnsOptionalValues,
      typename reader_t::exec_in_t,
      typename reader_t::exec_null_free_in_t>::type;

  ArrayView(const reader_t* reader, vector_size_t offset, vector_size_t size)
      : reader_(reader), offset_(offset), size_(size) {}

  using Element = typename std::
      conditional<returnsOptionalValues, OptionalAccessor<V>, element_t>::type;

  class ElementAccessor {
   public:
    using element_t = Element;
    using index_t = vector_size_t;

    explicit ElementAccessor(const reader_t* reader) : reader_(reader) {}

    Element operator()(vector_size_t index) const {
      if constexpr (returnsOptionalValues) {
        return Element{reader_, index};
      } else {
        return reader_->readNullFree(index);
      }
    }

   private:
    const reader_t* reader_;
  };

  using Iterator = IndexBasedIterator<ElementAccessor>;

  Iterator begin() const {
    return Iterator{
        offset_, offset_, offset_ + size_, ElementAccessor(reader_)};
  }

  Iterator end() const {
    return Iterator{
        offset_ + size_, offset_, offset_ + size_, ElementAccessor(reader_)};
  }

  bool empty() const {
    return size() == 0;
  }
};

}


// velox/expression/VectorReaders.h
namespace facebook::velox::exec {

template <typename T>
struct VectorReader;

template <typename V>
struct VectorReader<Array<V>> {
  using exec_in_t = typename VectorExec::template resolver<Array<V>>::in_type;
  using exec_null_free_in_t =
      typename VectorExec::template resolver<Array<V>>::null_free_in_type;
  using exec_in_child_t = typename VectorExec::template resolver<V>::in_type;

  explicit VectorReader(const DecodedVector* decoded)
      : decoded_(*decoded),
        vector_(detail::getDecoded<ArrayVector>(decoded_)),
        offsets_{vector_.rawOffsets()},
        lengths_{vector_.rawSizes()},
        childReader_{detail::decode(arrayValuesDecoder_, *vector_.elements())} {
  }

  bool isSet(size_t offset) const {
    return !decoded_.isNullAt(offset);
  }

  exec_in_t operator[](size_t offset) const {
    auto index = decoded_.index(offset);
    return {&childReader_, offsets_[index], lengths_[index]};
  }

  exec_null_free_in_t readNullFree(size_t offset) const {
    auto index = decoded_.index(offset);
    return {&childReader_, offsets_[index], lengths_[index]};
  }

  DecodedVector arrayValuesDecoder_;
  const DecodedVector& decoded_;
  const ArrayVector& vector_;
  const vector_size_t* offsets_;
  const vector_size_t* lengths_;
  VectorReader<V> childReader_;
};

}