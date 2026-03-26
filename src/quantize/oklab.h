#pragma once

/// \file oklab.h
/// \brief sRGB <-> OKLab color space conversions (header-only).
///
/// Based on Björn Ottosson's OKLab specification:
///   sRGB -> linear RGB -> M1 matrix -> cube root -> M2 matrix -> OKLab
///
/// Reference: https://bottosson.github.io/posts/oklab/

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace neroued::vectorizer::detail {

struct OkLab {
    float L = 0.f; ///< Lightness [0, 1].
    float a = 0.f; ///< Green–red axis.
    float b = 0.f; ///< Blue–yellow axis.
};

namespace oklab_internal {

inline const float* GetSrgbToLinearLUT() {
    static float lut[256] = {};
    static bool ready     = false;
    if (!ready) {
        for (int i = 0; i < 256; ++i) {
            float s = static_cast<float>(i) / 255.f;
            lut[i]  = (s <= 0.04045f) ? s / 12.92f : std::pow((s + 0.055f) / 1.055f, 2.4f);
        }
        ready = true;
    }
    return lut;
}

inline float SrgbToLinearFast(uint8_t v) { return GetSrgbToLinearLUT()[v]; }

inline float SrgbToLinear(float c) {
    return (c <= 0.04045f) ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

inline float LinearToSrgb(float c) {
    return (c <= 0.0031308f) ? 12.92f * c : 1.055f * std::pow(c, 1.f / 2.4f) - 0.055f;
}

inline OkLab LinearRgbToOklab(float r, float g, float b) {
    float l = 0.4122214708f * r + 0.5363325363f * g + 0.0514459929f * b;
    float m = 0.2119034982f * r + 0.6806995451f * g + 0.1073969566f * b;
    float s = 0.0883024619f * r + 0.2817188376f * g + 0.6299787005f * b;

    l = std::cbrt(l);
    m = std::cbrt(m);
    s = std::cbrt(s);

    OkLab lab;
    lab.L = 0.2104542553f * l + 0.7936177850f * m - 0.0040720468f * s;
    lab.a = 1.9779984951f * l - 2.4285922050f * m + 0.4505937099f * s;
    lab.b = 0.0259040371f * l + 0.7827717662f * m - 0.8086757660f * s;
    return lab;
}

} // namespace oklab_internal

inline OkLab SrgbToOklab(uint8_t r8, uint8_t g8, uint8_t b8) {
    return oklab_internal::LinearRgbToOklab(oklab_internal::SrgbToLinearFast(r8),
                                            oklab_internal::SrgbToLinearFast(g8),
                                            oklab_internal::SrgbToLinearFast(b8));
}

inline OkLab SrgbToOklab(float r01, float g01, float b01) {
    return oklab_internal::LinearRgbToOklab(oklab_internal::SrgbToLinear(r01),
                                            oklab_internal::SrgbToLinear(g01),
                                            oklab_internal::SrgbToLinear(b01));
}

inline void OklabToSrgb(const OkLab& lab, uint8_t& r8, uint8_t& g8, uint8_t& b8) {
    float l = lab.L + 0.3963377774f * lab.a + 0.2158037573f * lab.b;
    float m = lab.L - 0.1055613458f * lab.a - 0.0638541728f * lab.b;
    float s = lab.L - 0.0894841775f * lab.a - 1.2914855480f * lab.b;

    l = l * l * l;
    m = m * m * m;
    s = s * s * s;

    float r = +4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s;
    float g = -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s;
    float b = -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s;

    r = std::clamp(oklab_internal::LinearToSrgb(std::clamp(r, 0.f, 1.f)), 0.f, 1.f);
    g = std::clamp(oklab_internal::LinearToSrgb(std::clamp(g, 0.f, 1.f)), 0.f, 1.f);
    b = std::clamp(oklab_internal::LinearToSrgb(std::clamp(b, 0.f, 1.f)), 0.f, 1.f);

    r8 = static_cast<uint8_t>(std::lround(r * 255.f));
    g8 = static_cast<uint8_t>(std::lround(g * 255.f));
    b8 = static_cast<uint8_t>(std::lround(b * 255.f));
}

} // namespace neroued::vectorizer::detail
