#pragma once

/// \file vec2.h
/// \brief 2D floating-point vector.

#include <cmath>

namespace neroued::vectorizer {

struct Vec2f {
    float x = 0.0f;
    float y = 0.0f;

    Vec2f() = default;

    constexpr Vec2f(float x, float y) : x(x), y(y) {}

    Vec2f operator+(const Vec2f& o) const { return {x + o.x, y + o.y}; }

    Vec2f operator-(const Vec2f& o) const { return {x - o.x, y - o.y}; }

    Vec2f operator*(float s) const { return {x * s, y * s}; }

    Vec2f operator/(float s) const { return {x / s, y / s}; }

    Vec2f& operator+=(const Vec2f& o) {
        x += o.x;
        y += o.y;
        return *this;
    }

    Vec2f& operator-=(const Vec2f& o) {
        x -= o.x;
        y -= o.y;
        return *this;
    }

    Vec2f& operator*=(float s) {
        x *= s;
        y *= s;
        return *this;
    }

    Vec2f& operator/=(float s) {
        x /= s;
        y /= s;
        return *this;
    }

    bool operator==(const Vec2f& o) const { return x == o.x && y == o.y; }

    bool operator!=(const Vec2f& o) const { return !(*this == o); }

    float Dot(const Vec2f& o) const { return x * o.x + y * o.y; }

    float Cross(const Vec2f& o) const { return x * o.y - y * o.x; }

    float LengthSquared() const { return Dot(*this); }

    float Length() const { return std::sqrt(LengthSquared()); }

    Vec2f Normalized() const {
        float len = Length();
        return len > 0.0f ? (*this / len) : Vec2f();
    }

    static Vec2f Lerp(const Vec2f& a, const Vec2f& b, float t) { return a + (b - a) * t; }

    static float Distance(const Vec2f& a, const Vec2f& b) { return (a - b).Length(); }
};

inline Vec2f operator*(float s, const Vec2f& v) { return v * s; }

} // namespace neroued::vectorizer
