@echo off
setlocal

REM ── CI dependency installer for Windows ─────────────────────────────────────
REM Uses VCPKG_INSTALLATION_ROOT provided by GitHub Actions runners.
REM Falls back to cloning vcpkg if the variable is not set.

if defined VCPKG_INSTALLATION_ROOT (
    set "VCPKG=%VCPKG_INSTALLATION_ROOT%\vcpkg"
) else if exist "C:\vcpkg\vcpkg.exe" (
    set "VCPKG=C:\vcpkg\vcpkg"
) else (
    echo Cloning vcpkg ...
    git clone --depth 1 https://github.com/microsoft/vcpkg.git C:\vcpkg
    call C:\vcpkg\bootstrap-vcpkg.bat -disableMetrics
    set "VCPKG=C:\vcpkg\vcpkg"
)

echo Installing OpenCV and Potrace via vcpkg ...
"%VCPKG%" install opencv4:x64-windows potrace:x64-windows

echo === Windows dependency installation complete ===
