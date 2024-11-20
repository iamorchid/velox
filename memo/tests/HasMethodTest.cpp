#include <stdint.h>
#include <iostream>
#include <type_traits>
#include <utility>

struct Example {
  // C++14允许使用尾置返回类型（trailing return type）来指示返回类型，
  // 这在复杂的情况下特别有用，例如当函数的返回类型取决于条件表达式或其他
  // 复杂逻辑时。
  template <typename T1, typename T2>
  auto add(T1 a, T2 b) -> decltype(a + b) {
    return a + b;
  }
};

void TestExampleAdd() {
  // std::declval是C++中的一个实用函数模板，它在不需要创建对象实例的情况下，
  // 允许你获得一个给定类型的引用。这在模板元编程和SFINAE（Substitution
  // Failure Is Not An Error）场景中特别有用，尤其是当你希望推断某个表达
  // 式的类型或者检查类型是否存在某个成员函数时。
  using ValueType =
      decltype(std::declval<Example>().add<int64_t, int64_t>(20, 10));

  // 现在ValueType是int类型，因为Example::value()返回int
  static_assert(
      std::is_same<ValueType, int64_t>::value, "ValueType should be int64_t");
}

#define DECLARE_METHOD_RESOLVER(Name, MethodName)              \
  struct Name {                                                \
    template <class __T, typename... __TArgs>                  \
    constexpr auto resolve(__TArgs&&... args) const            \
        -> decltype(std::declval<__T>().MethodName(args...)) { \
      return {};                                               \
    }                                                          \
  };

template <typename C, class TResolver, typename TRet, typename... TArgs>
struct has_method {
 private:
  template <typename T>
  static constexpr auto check(T*) -> typename std::is_same<
      decltype(std::declval<TResolver>().template resolve<T>(
          std::declval<TArgs>()...)),
      TRet>::type {
    return {};
  }

  template <typename>
  static constexpr std::false_type check(...) {
    return std::false_type();
  }

  using type = decltype(check<C>(nullptr));

 public:
  static constexpr bool value = type::value;
};

DECLARE_METHOD_RESOLVER(add_method_resolver, add);

int main() {
  constexpr bool hasAdd =
      has_method<Example, add_method_resolver, int32_t, int32_t, int32_t>::
          value;
  std::cout << "hasAdd: " << hasAdd << std::endl;
  return 0;
}
