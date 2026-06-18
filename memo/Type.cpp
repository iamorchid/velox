#include <stdint.h>
#include <string>

// velox/type/Type.h
namespace facebook::velox {

enum class TypeKind : int8_t {
  BOOLEAN = 0,
  TINYINT = 1,
  SMALLINT = 2,
  INTEGER = 3,
  BIGINT = 4,
  REAL = 5,
  DOUBLE = 6,
  VARCHAR = 7,
  VARBINARY = 8,
  TIMESTAMP = 9,
  HUGEINT = 10,
  // Enum values for ComplexTypes start after 30 to leave
  // some values space to accommodate adding new scalar/native
  // types above.
  ARRAY = 30,
  MAP = 31,
  ROW = 32,
  UNKNOWN = 33,
  FUNCTION = 34,
  OPAQUE = 35,
  INVALID = 36
};

template <TypeKind KIND>
struct TypeTraits {};

template <>
struct TypeTraits<TypeKind::BIGINT> {
  using ImplType = ScalarType<TypeKind::BIGINT>;
  using NativeType = int64_t;
  using DeepCopiedType = NativeType;
  static constexpr uint32_t minSubTypes = 0;
  static constexpr uint32_t maxSubTypes = 0;
  static constexpr TypeKind typeKind = TypeKind::BIGINT;
  static constexpr bool isPrimitiveType = true;
  static constexpr bool isFixedWidth = true;
  static constexpr const char* name = "BIGINT";
};


template <>
struct TypeTraits<TypeKind::VARCHAR> {
  using ImplType = ScalarType<TypeKind::VARCHAR>;
  using NativeType = velox::StringView;
  using DeepCopiedType = std::string;
  static constexpr uint32_t minSubTypes = 0;
  static constexpr uint32_t maxSubTypes = 0;
  static constexpr TypeKind typeKind = TypeKind::VARCHAR;
  static constexpr bool isPrimitiveType = true;
  static constexpr bool isFixedWidth = false;
  static constexpr const char* name = "VARCHAR";
};

template <typename T>
struct CppToType {};

template <TypeKind KIND>
struct CppToTypeBase : public TypeTraits<KIND> {
  static auto create() {
    return TypeFactory<KIND>::create();
  }
};

template <>
struct CppToType<int64_t> : public CppToTypeBase<TypeKind::BIGINT> {};


template <>
struct CppToType<velox::StringView> : public CppToTypeBase<TypeKind::VARCHAR> {
};

template <>
struct CppToType<std::string> : public CppToTypeBase<TypeKind::VARCHAR> {};


template <>
struct CppToType<const char*> : public CppToTypeBase<TypeKind::VARCHAR> {};


template <>
struct CppToType<Varbinary> : public CppToTypeBase<TypeKind::VARBINARY> {};


template <typename T>
struct SimpleTypeTrait {};

template <>
struct SimpleTypeTrait<int64_t> : public TypeTraits<TypeKind::BIGINT> {};

template <>
struct SimpleTypeTrait<Varchar> : public TypeTraits<TypeKind::VARCHAR> {};

}
