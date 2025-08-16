#pragma once

#include <any>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

namespace stringanyting {
// ========== 异常 ==========
struct KeyNotFound: public std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct BadType: public std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct AlreadyDefined: public std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct TimeoutError: public std::runtime_error {
    using std::runtime_error::runtime_error;
};

// ========== 静态 Holder（用于以 shared_ptr 形式存放可拷贝值） ==========
class IStaticHolder {
public:
    virtual ~IStaticHolder() = default;
    virtual std::type_index type() const = 0;
};

template<typename T>
class StaticHolder: public IStaticHolder {
public:
    explicit StaticHolder(T v): value_(std::move(v)) {}
    std::type_index type() const override {
        return typeid(T);
    }
    T get_copy() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return value_;
    }
    void set(T v) {
        std::lock_guard<std::mutex> lk(mtx_);
        value_ = std::move(v);
    }

private:
    mutable std::mutex mtx_;
    T value_;
};

// ========== 管理器：按字符串存储 value 或 shared_ptr<T> ==========
class Manager {
public:
    using sptr = std::shared_ptr<Manager>;
    static sptr create() {
        return std::make_shared<Manager>();
    }

    Manager() = default;

    enum class Kind { Value, Ptr };

    // ---- set_value: 存放一个值（线程通过 get_value 拿到拷贝） ----
    // overwrite: 是否允许覆盖已存在条目（若已存在类型不匹配抛 BadType）
    template<typename T>
    void set_value(const std::string& name, T value, bool overwrite = true) {
        std::lock_guard<std::mutex> lk(map_mtx_);
        auto it = map_.find(name);
        if (it != map_.end()) {
            if (it->second.kind != Kind::Value)
                throw BadType("existing entry is Ptr, not Value: " + name);
            if (it->second.type != std::type_index(typeid(T)))
                throw BadType("type mismatch for value entry: " + name);
            if (!overwrite)
                throw AlreadyDefined("value exists: " + name);
            // 替换 holder 中的值
            auto hold = std::any_cast<std::shared_ptr<StaticHolder<T>>>(it->second.storage);
            hold->set(std::move(value));
            return;
        }
        // 新建 holder 并存入 any
        std::shared_ptr<StaticHolder<T>> holder =
            std::make_shared<StaticHolder<T>>(std::move(value));
        Entry e;
        e.kind = Kind::Value;
        e.type = std::type_index(typeid(T));
        e.storage = std::any(holder);
        map_.emplace(name, std::move(e));
    }

    // ---- get_value: 返回拷贝 ----
    template<typename T>
    T get_value(const std::string& name) const {
        std::lock_guard<std::mutex> lk(map_mtx_);
        auto it = map_.find(name);
        if (it == map_.end())
            throw KeyNotFound("value not found: " + name);
        const Entry& e = it->second;
        if (e.kind != Kind::Value)
            throw BadType("entry is Ptr, not Value: " + name);
        if (e.type != std::type_index(typeid(T)))
            throw BadType("value type mismatch: " + name);
        auto holder = std::any_cast<std::shared_ptr<StaticHolder<T>>>(e.storage);
        return holder->get_copy();
    }

    // ---- set_ptr: 存放 shared_ptr<T> ----
    // overwrite 参数与上类似
    template<typename T>
    void set_ptr(const std::string& name, std::shared_ptr<T> p, bool overwrite = true) {
        std::lock_guard<std::mutex> lk(map_mtx_);
        auto it = map_.find(name);
        if (it != map_.end()) {
            if (it->second.kind != Kind::Ptr)
                throw BadType("existing entry is Value: " + name);
            if (it->second.type != std::type_index(typeid(T)))
                throw BadType("type mismatch: " + name);
            if (!overwrite)
                throw AlreadyDefined("ptr exists: " + name);
            it->second.storage = std::any(p); // ✅ 直接存 shared_ptr<T>
            return;
        }
        Entry e;
        e.kind = Kind::Ptr;
        e.type = std::type_index(typeid(T));
        e.storage = std::any(p); // ✅ 直接存 shared_ptr<T>
        map_.emplace(name, std::move(e));
    }

    template<typename T>
    std::shared_ptr<T> get_ptr(const std::string& name) const {
        std::lock_guard<std::mutex> lk(map_mtx_);
        auto it = map_.find(name);
        if (it == map_.end())
            throw KeyNotFound("ptr not found: " + name);
        const Entry& e = it->second;
        if (e.kind != Kind::Ptr)
            throw BadType("entry is Value: " + name);
        if (e.type != std::type_index(typeid(T)))
            throw BadType("type mismatch: " + name);
        return std::any_cast<std::shared_ptr<T>>(e.storage); // ✅ 与存储类型匹配
    }

    // ---- create_ptr: factory 创建并返回 shared_ptr<T>, 若已存在返回现有（可选择覆盖） ----
    template<typename T>
    std::shared_ptr<T> create_ptr(
        const std::string& name,
        std::function<std::shared_ptr<T>()> factory,
        bool overwrite = false
    ) {
        {
            std::lock_guard<std::mutex> lk(map_mtx_);
            auto it = map_.find(name);
            if (it != map_.end()) {
                if (it->second.kind != Kind::Ptr)
                    throw BadType("existing entry kind mismatch: " + name);
                if (it->second.type != std::type_index(typeid(T)))
                    throw BadType("existing entry type mismatch: " + name);
                // 返回已有对象
                return std::any_cast<std::shared_ptr<void>>(it->second.storage
                ); // will throw if wrong type
            }
        }
        auto p = factory();
        set_ptr<T>(name, p, true);
        return p;
    }

    // ---- exists / erase / names ----
    bool exists(const std::string& name) const {
        std::lock_guard<std::mutex> lk(map_mtx_);
        return map_.find(name) != map_.end();
    }

    void erase(const std::string& name) {
        std::lock_guard<std::mutex> lk(map_mtx_);
        map_.erase(name);
    }

    std::vector<std::string> names() const {
        std::lock_guard<std::mutex> lk(map_mtx_);
        std::vector<std::string> out;
        out.reserve(map_.size());
        for (const auto& kv: map_)
            out.push_back(kv.first);
        return out;
    }

private:
    struct Entry {
        Kind kind = Kind::Value;
        std::type_index type = std::type_index(typeid(void)); // 初始化，避免无默认构造
        std::any storage;

        Entry() noexcept: kind(Kind::Value), type(std::type_index(typeid(void))), storage() {}

        Entry(Kind k, std::type_index t, std::any s): kind(k), type(t), storage(std::move(s)) {}
    };

    mutable std::mutex map_mtx_;
    std::unordered_map<std::string, Entry> map_;
};

} // namespace stringanyting
