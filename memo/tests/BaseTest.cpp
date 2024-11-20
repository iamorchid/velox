#include <iostream>
#include <memory>
#include <string>

class TestBase {
 public:
  virtual ~TestBase() = default;

  virtual std::string getName() = 0;

  template <typename T>
  T* as() {
    static_assert(std::is_base_of_v<TestBase, T>);
    return dynamic_cast<T*>(this);
  }
};

using TestBasePtr = std::shared_ptr<TestBase>;

template <class T>
class InregralTest : public TestBase {
 public:
  InregralTest(T v) : val(v) {}

  std::string getName() override {
    return "InregralTest";
  }

  T getValue() const {
    return val;
  }

 private:
  T val;
};

int main() {
  TestBasePtr testPtr = std::make_shared<InregralTest<int32_t>>(0x10001000);
  std::cout << testPtr->getName() << std::endl;

  auto* integral32Test = testPtr->template as<InregralTest<int32_t>>();
  std::cout << integral32Test->getValue() << std::endl;

  // integral16Test将为nullptr
  auto* integral16Test = testPtr->template as<InregralTest<int16_t>>();
  std::cout << integral16Test << std::endl;

  return 0;
}