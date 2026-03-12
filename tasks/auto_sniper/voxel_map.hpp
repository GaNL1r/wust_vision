#pragma once

#include "3rdparty/ankerl/unordered_dense.h"
#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <vector>
namespace wust_vision::auto_sniper {

template<int Dim>
struct VoxelKey {
    std::array<int, Dim> data {};

    int& operator[](int i) noexcept {
        return data[i];
    }
    const int& operator[](int i) const noexcept {
        return data[i];
    }

    // ----- x -----
    int& x() noexcept {
        static_assert(Dim >= 1, "x requires Dim >= 1");
        return data[0];
    }
    const int& x() const noexcept {
        static_assert(Dim >= 1, "x requires Dim >= 1");
        return data[0];
    }

    // ----- y -----
    int& y() noexcept {
        static_assert(Dim >= 2, "y requires Dim >= 2");
        return data[1];
    }
    const int& y() const noexcept {
        static_assert(Dim >= 2, "y requires Dim >= 2");
        return data[1];
    }

    // ----- z -----
    int& z() noexcept {
        static_assert(Dim >= 3, "z requires Dim >= 3");
        return data[2];
    }
    const int& z() const noexcept {
        static_assert(Dim >= 3, "z requires Dim >= 3");
        return data[2];
    }
    bool operator==(const VoxelKey& other) const noexcept {
        for (int i = 0; i < Dim; ++i)
            if (data[i] != other.data[i])
                return false;
        return true;
    }

    bool operator!=(const VoxelKey& other) const noexcept {
        return !(*this == other);
    }

    VoxelKey& operator+=(const VoxelKey& other) noexcept {
        for (int i = 0; i < Dim; ++i)
            data[i] += other.data[i];
        return *this;
    }

    VoxelKey& operator-=(const VoxelKey& other) noexcept {
        for (int i = 0; i < Dim; ++i)
            data[i] -= other.data[i];
        return *this;
    }

    VoxelKey& operator*=(int scalar) noexcept {
        for (int i = 0; i < Dim; ++i)
            data[i] *= scalar;
        return *this;
    }

    VoxelKey& operator/=(int scalar) noexcept {
        for (int i = 0; i < Dim; ++i)
            data[i] /= scalar;
        return *this;
    }

    friend VoxelKey operator+(VoxelKey lhs, const VoxelKey& rhs) noexcept {
        lhs += rhs;
        return lhs;
    }

    friend VoxelKey operator-(VoxelKey lhs, const VoxelKey& rhs) noexcept {
        lhs -= rhs;
        return lhs;
    }

    friend VoxelKey operator*(VoxelKey lhs, int scalar) noexcept {
        lhs *= scalar;
        return lhs;
    }

    friend VoxelKey operator*(int scalar, VoxelKey rhs) noexcept {
        rhs *= scalar;
        return rhs;
    }

    friend VoxelKey operator/(VoxelKey lhs, int scalar) noexcept {
        lhs /= scalar;
        return lhs;
    }
    constexpr VoxelKey cwiseMin(const VoxelKey& other) const noexcept {
        VoxelKey out {};
        for (int i = 0; i < Dim; ++i)
            out.data[i] = data[i] < other.data[i] ? data[i] : other.data[i];
        return out;
    }

    constexpr VoxelKey cwiseMax(const VoxelKey& other) const noexcept {
        VoxelKey out {};
        for (int i = 0; i < Dim; ++i)
            out.data[i] = data[i] > other.data[i] ? data[i] : other.data[i];
        return out;
    }
};

template<int Dim, typename Cell>
class SlidingVoxelMap {
    static_assert(Dim == 2 || Dim == 3, "Dim must be 2 or 3");

public:
    using Ptr = std::shared_ptr<SlidingVoxelMap>;
    using Key = VoxelKey<Dim>;
    using EigenPoint = Eigen::Matrix<double, Dim, 1>;

    SlidingVoxelMap(double voxel_size_, const EigenPoint& size_, const EigenPoint& center_):
        voxel_size(voxel_size_),
        size(size_),
        center(center_) {
        EigenPoint half = size * 0.5f;

        min_key = worldToKey(center - half);
        max_key = worldToKey(center + half);

        for (int i = 0; i < Dim; ++i) {
            dims[i] = max_key[i] - min_key[i] + 1;
            offset[i] = 0;
        }

        center_key = worldToKey(center);

        size_t N = 1;
        for (int i = 0; i < Dim; ++i)
            N *= static_cast<size_t>(dims[i]);

        grid.resize(N);
    }
    SlidingVoxelMap(
        double voxel_size_,
        const EigenPoint& min_pos,
        const EigenPoint& max_pos,
        bool /*dummy*/
    ):
        voxel_size(voxel_size_) {
        min_key = worldToKey(min_pos);
        max_key = worldToKey(max_pos);

        for (int i = 0; i < Dim; ++i) {
            if (max_key[i] < min_key[i])
                std::swap(max_key[i], min_key[i]);

            dims[i] = max_key[i] - min_key[i] + 1;
            offset[i] = 0;
        }

        for (int i = 0; i < Dim; ++i)
            center_key[i] = (min_key[i] + max_key[i]) / 2;

        EigenPoint min_world = keyToWorld(min_key);
        EigenPoint max_world = keyToWorld(max_key);

        for (int i = 0; i < Dim; ++i) {
            center[i] = (min_world[i] + max_world[i]) * 0.5f;
            size[i] = dims[i] * voxel_size;
        }

        size_t N = 1;
        for (int i = 0; i < Dim; ++i)
            N *= static_cast<size_t>(dims[i]);

        grid.resize(N);
    }

    static Ptr create(double voxel_size, const EigenPoint& size, const EigenPoint& center) {
        return std::make_shared<SlidingVoxelMap>(voxel_size, size, center);
    }

    size_t gridSize() const noexcept {
        return grid.size();
    }
    inline int worldToIndex(const EigenPoint& p) const noexcept {
        return keyToIndex(worldToKey(p));
    }

    inline Key worldToKey(const EigenPoint& p) const noexcept {
        Key k;
        const double inv = 1.0f / voxel_size;
        for (int i = 0; i < Dim; ++i)
            k.data[i] = static_cast<int>(std::floor(p[i] * inv + 1e-6f));
        return k;
    }

    inline EigenPoint keyToWorld(const Key& k) const noexcept {
        EigenPoint p;
        for (int i = 0; i < Dim; ++i)
            p[i] = (k.data[i] + 0.5f) * voxel_size;
        return p;
    }

    inline EigenPoint indexToWorld(int idx) const noexcept {
        return keyToWorld(indexToKey(idx));
    }

    inline int keyToIndex(const Key& k) const noexcept {
        int idx = 0;
        int stride = 1;

        for (int d = Dim - 1; d >= 0; --d) {
            int delta = k[d] - center_key[d] + (dims[d] >> 1);

            if (delta < 0 || delta >= dims[d])
                return -1;

            int r = delta + offset[d];

            if (r >= dims[d])
                r -= dims[d];
            else if (r < 0)
                r += dims[d];

            idx += r * stride;
            stride *= dims[d];
        }

        return idx;
    }

    inline Key indexToKey(int idx) const noexcept {
        Key k;

        for (int d = Dim - 1; d >= 0; --d) {
            int r = idx % dims[d];
            idx /= dims[d];

            int delta = r - offset[d];

            if (delta < 0)
                delta += dims[d];
            else if (delta >= dims[d])
                delta -= dims[d];

            k[d] = center_key[d] + delta - (dims[d] >> 1);
        }

        return k;
    }
    template<typename ClearFunc>
    void slideTo(const Key& new_center_key, ClearFunc clear_func) {
        Key shift;
        for (int d = 0; d < Dim; ++d)
            shift[d] = new_center_key[d] - center_key[d];

        for (int d = 0; d < Dim; ++d) {
            if (std::abs(shift[d]) >= dims[d]) {
                for (size_t i = 0; i < grid.size(); ++i)
                    clear_func(i);

                offset = {};
                center_key = new_center_key;
                return;
            }
        }
        for (int axis = 0; axis < Dim; ++axis) {
            int s = shift[axis];
            if (s == 0)
                continue;

            int steps = std::abs(s);
            int dir = s > 0 ? 1 : -1;

            for (int step = 0; step < steps; ++step) {
                int slice = (offset[axis] + dir * step + dims[axis]) % dims[axis];

                clearSlice(axis, slice, clear_func);
            }

            offset[axis] = (offset[axis] + s + dims[axis]) % dims[axis];
        }

        center_key = new_center_key;
        EigenPoint half = size * 0.5f;
        center = keyToWorld(center_key);
        min_key = worldToKey(center - half);
        max_key = worldToKey(center + half);
    }

    template<typename ClearFunc>
    void clearSlice(int axis, int slice, ClearFunc clear_func) {
        if constexpr (Dim == 3) {
            int dx = dims[0];
            int dy = dims[1];
            int dz = dims[2];

            if (axis == 0) {
                for (int y = 0; y < dy; ++y)
                    for (int z = 0; z < dz; ++z) {
                        int idx = (slice * dy + y) * dz + z;
                        clear_func(idx);
                    }
            } else if (axis == 1) {
                for (int x = 0; x < dx; ++x)
                    for (int z = 0; z < dz; ++z) {
                        int idx = (x * dy + slice) * dz + z;
                        clear_func(idx);
                    }
            } else {
                for (int x = 0; x < dx; ++x)
                    for (int y = 0; y < dy; ++y) {
                        int idx = (x * dy + y) * dz + slice;
                        clear_func(idx);
                    }
            }
        }
    }

public:
    double voxel_size;

    Key dims;
    Key offset;

    std::vector<Cell> grid;

    Key center_key;
    EigenPoint center;
    EigenPoint size;
    Key min_key;
    Key max_key;
};

} // namespace wust_vision::auto_sniper