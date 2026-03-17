#!/bin/bash
set -euo pipefail

# ── CI dependency installer for manylinux_2_28 (AlmaLinux 8) ─────────────────
#
# Dependency versions — keep in sync with install-deps-windows.bat
OPENCV_VER="4.9.0"
POTRACE_VER="1.16"

yum install -y cmake gcc gcc-c++ make git curl

# ── OpenCV ────────────────────────────────────────────────────────────────────
install_opencv_from_source() {
    echo "Building OpenCV ${OPENCV_VER} from source ..."
    yum install -y libpng-devel libjpeg-turbo-devel libtiff-devel libwebp-devel zlib-devel
    cd /tmp
    curl -L -o opencv.tar.gz \
        "https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VER}.tar.gz"
    tar xzf opencv.tar.gz
    cmake -S "opencv-${OPENCV_VER}" -B opencv-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_LIST=core,imgproc,imgcodecs \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_opencv_apps=OFF \
        -DWITH_FFMPEG=OFF \
        -DWITH_GTK=OFF \
        -DWITH_V4L=OFF \
        -DWITH_OPENCL=OFF
    cmake --build opencv-build -j"$(nproc)"
    cmake --install opencv-build
    ldconfig
    rm -rf /tmp/opencv*
}

yum install -y epel-release || true
yum config-manager --set-enabled powertools 2>/dev/null \
    || dnf config-manager --set-enabled powertools 2>/dev/null \
    || true

install_opencv_from_source

# ── Potrace ───────────────────────────────────────────────────────────────────
install_potrace_from_source() {
    echo "Building potrace ${POTRACE_VER} from source ..."
    cd /tmp
    curl -L -o potrace.tar.gz \
        "https://potrace.sourceforge.net/download/${POTRACE_VER}/potrace-${POTRACE_VER}.tar.gz"
    tar xzf potrace.tar.gz
    cd "potrace-${POTRACE_VER}"
    ./configure --with-libpotrace --prefix=/usr/local
    make -j"$(nproc)"
    make install
    ldconfig
    rm -rf /tmp/potrace*
}

install_potrace_from_source

echo "=== Linux dependency installation complete ==="
