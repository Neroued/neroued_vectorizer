#pragma once

/// \file vec3.h
/// \brief 3-component integer and float vector types.

#include <algorithm>
#include <cmath>

namespace neroued::vectorizer {

/// 3-component integer vector.
struct Vec3i {
    int x = 0;
    int y = 0;
    int z = 0;

    constexpr Vec3i() = default;

    constexpr Vec3i(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}

    Vec3i operator+(const Vec3i& o) const { return {x + o.x, y + o.y, z + o.z}; }

    Vec3i operator-(const Vec3i& o) const { return {x - o.x, y - o.y, z - o.z}; }

    Vec3i operator*(int s) const { return {x * s, y * s, z * s}; }

    Vec3i operator/(int s) const { return {x / s, y / s, z / s}; }

    Vec3i& operator+=(const Vec3i& o) {
        x += o.x;
        y += o.y;
        z += o.z;
        return *this;
    }

    Vec3i& operator-=(const Vec3i& o) {
        x -= o.x;
        y -= o.y;
        z -= o.z;
        return *this;
    }

    Vec3i& operator*=(int s) {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    Vec3i& operator/=(int s) {
        x /= s;
        y /= s;
        z /= s;
        return *this;
    }

    int& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }

    const int& operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }

    int Dot(const Vec3i& o) const { return x * o.x + y * o.y + z * o.z; }

    int LengthSquared() const { return Dot(*this); }
};

inline Vec3i operator*(int s, const Vec3i& v) { return v * s; }

/// 3-component float vector.
struct Vec3f {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    constexpr Vec3f() = default;

    constexpr Vec3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    Vec3f operator+(const Vec3f& o) const { return {x + o.x, y + o.y, z + o.z}; }

    Vec3f operator-(const Vec3f& o) const { return {x - o.x, y - o.y, z - o.z}; }

    Vec3f operator*(float s) const { return {x * s, y * s, z * s}; }

    Vec3f operator/(float s) const { return {x / s, y / s, z / s}; }

    Vec3f& operator+=(const Vec3f& o) {
        x += o.x;
        y += o.y;
        z += o.z;
        return *this;
    }

    Vec3f& operator-=(const Vec3f& o) {
        x -= o.x;
        y -= o.y;
        z -= o.z;
        return *this;
    }

    Vec3f& operator*=(float s) {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    Vec3f& operator/=(float s) {
        x /= s;
        y /= s;
        z /= s;
        return *this;
    }

    float& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }

    const float& operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }

    float Dot(const Vec3f& o) const { return x * o.x + y * o.y + z * o.z; }

    float LengthSquared() const { return Dot(*this); }

    float Length() const { return std::sqrt(LengthSquared()); }

    Vec3f Normalized() const {
        float len = Length();
        return len > 0.0f ? (*this / len) : Vec3f();
    }

    bool IsFinite() const { return std::isfinite(x) && std::isfinite(y) && std::isfinite(z); }

    bool NearlyEqual(const Vec3f& o, float eps = 1e-5f) const {
        return std::fabs(x - o.x) <= eps && std::fabs(y - o.y) <= eps && std::fabs(z - o.z) <= eps;
    }

    static Vec3f Lerp(const Vec3f& a, const Vec3f& b, float t) { return a + (b - a) * t; }

    static Vec3f Clamp(const Vec3f& v, float lo, float hi) {
        auto c = [](float v, float lo, float hi) { return std::max(lo, std::min(v, hi)); };
        return {c(v.x, lo, hi), c(v.y, lo, hi), c(v.z, lo, hi)};
    }

    static Vec3f Clamp01(const Vec3f& v) { return Clamp(v, 0.0f, 1.0f); }

    static float Distance(const Vec3f& a, const Vec3f& b) { return (a - b).Length(); }
};

inline Vec3f operator*(float s, const Vec3f& v) { return v * s; }

} // namespace neroued::vectorizer
