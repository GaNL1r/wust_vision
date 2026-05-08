#ifndef FACTORY_H
#define FACTORY_H

#include <concepts>
#include <unordered_map>

#define enable_factory(_namespace, _type, _func)                                                             \
  namespace _namespace {                                                                                     \
  class _type;                                                                                               \
  inline _type *_func(const std::string &type_name) { return Factory<_type>::Instance().Create(type_name); } \
  }

namespace wust_vision {

template <class B>
class RegistryBase {
 public:
  virtual ~RegistryBase() = default;
  virtual B *Create() = 0;
};

template <class B>
class Factory final {
 public:
  [[nodiscard]] static Factory &Instance() {
    static Factory factory;
    return factory;
  }

  void Register(std::string &&type_name, RegistryBase<B> *registry) { registry_[std::move(type_name)] = registry; }

  [[nodiscard]] B *Create(const std::string &type_name) {
    return registry_.contains(type_name) ? registry_[type_name]->Create() : nullptr;
  }

 private:
  Factory() = default;
  ~Factory() = default;
  std::unordered_map<std::string, RegistryBase<B> *> registry_;
};

template <class B, class S>
class RegistrySub final : public RegistryBase<B> {
 public:
  explicit RegistrySub(std::string &&type_name) {
    static_assert(std::is_base_of<B, S>());
    Factory<B>::Instance().Register(std::move(type_name), this);
  }

  [[nodiscard]] B *Create() override { return new S(); }
};
}  // namespace wust_vision
#endif //FACTORY_H