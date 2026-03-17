@echo off
setlocal enabledelayedexpansion

REM ── CI dependency installer for Windows ─────────────────────────────────────
REM
REM Dependency versions — keep in sync with install-deps-linux.sh
REM   OpenCV : installed via vcpkg (version determined by vcpkg port registry)
REM   Potrace: built from source
set POTRACE_VER=1.16
set POTRACE_PREFIX=C:\potrace

REM ── OpenCV via vcpkg ────────────────────────────────────────────────────────

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

echo Installing OpenCV via vcpkg (Release only, minimal features) ...
"%VCPKG%" install "opencv4[core,jpeg,png]:x64-windows-release" --host-triplet=x64-windows-release
if errorlevel 1 exit /b 1

REM ── Potrace from source (vcpkg has no port) ────────────────────────────────

echo Downloading potrace %POTRACE_VER% ...
curl -L -o potrace.tar.gz "https://potrace.sourceforge.net/download/%POTRACE_VER%/potrace-%POTRACE_VER%.tar.gz"
if errorlevel 1 exit /b 1
tar xzf potrace.tar.gz

echo Building potrace from source ...
copy /Y "%~dp0potrace-CMakeLists.txt" "potrace-%POTRACE_VER%\CMakeLists.txt"
cmake -S "potrace-%POTRACE_VER%" -B potrace-build -DCMAKE_INSTALL_PREFIX="%POTRACE_PREFIX%" -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1
cmake --build potrace-build --config Release
if errorlevel 1 exit /b 1
cmake --install potrace-build --config Release
if errorlevel 1 exit /b 1

echo === Windows dependency installation complete ===
