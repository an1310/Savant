#!/usr/bin/env bash
set -euo pipefail

# Build Savant's CUDA OpenCV package against the DeepStream image's CUDA SDK.
MODULE_DIR=${1:?OpenCV Savant module path required}
OUTPUT_FILE=${2:?Output tarball path required}
OPENCV_VERSION=${OPENCV_VERSION:-4.12.0}

/opt/nvidia/deepstream/deepstream/user_additional_install.sh
apt-get update
apt-get install --no-install-recommends -y \
    file libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
python3 -m pip install --no-cache-dir 'numpy>=1.22.4,<2.0'

mkdir -p /opencv
git clone --branch "$OPENCV_VERSION" --depth 1 \
    https://github.com/opencv/opencv /opencv/opencv
git clone --branch "$OPENCV_VERSION" --depth 1 \
    https://github.com/opencv/opencv_contrib /opencv/opencv_contrib
cp -a "$MODULE_DIR" /opencv/opencv_contrib/modules/savant
mkdir -p /opencv/build
cd /opencv/build

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
CUDA_OPTIONS=()
OPENCV_BUILD_LIST=core,cudaarithm,cudabgsegm,cudacodec,cudafeatures2d,cudafilters,cudaimgproc,cudev,features2d,flann,calib3d,imgcodecs,savant,python3
if [[ "${DEEPSTREAM_VERSION:-}" == "9.1" ]]; then
    # CUDA 13 no longer accepts Pascal or Volta targets. Explicit targets also
    # let this build run on a Linux builder without a visible GPU.
    CUDA_OPTIONS=(-D "CUDA_ARCH_BIN=7.5 8.0 8.6 8.9 9.0 12.0")
    # cudacodec still requires CUDA_CUDA_LIBRARY, which CUDA 13 no longer ships.
    OPENCV_BUILD_LIST=${OPENCV_BUILD_LIST/,cudacodec/}
fi
cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D OPENCV_EXTRA_MODULES_PATH=/opencv/opencv_contrib/modules \
    -D CMAKE_INSTALL_PREFIX=/opencv/dist \
    -D OPENCV_DOWNLOAD_PATH=/tmp/opencv-cache \
    -D PYTHON_DEFAULT_EXECUTABLE="$(command -v python3)" \
    -D BUILD_LIST="$OPENCV_BUILD_LIST" \
    -D BUILD_opencv_apps=OFF \
    -D BUILD_DOCS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_JAVA=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_SHARED_LIBS=ON \
    -D WITH_CUDA=ON \
    -D WITH_FFMPEG=ON \
    -D WITH_GSTREAMER=ON \
    -D BUILD_opencv_python3=ON \
    -D OPENCV_PYTHON_INSTALL_PATH="lib/python${PYTHON_VERSION}/dist-packages" \
    -D OPENCV_FORCE_PYTHON_LIBS=ON \
    "${CUDA_OPTIONS[@]}" \
    -D BUILD_PACKAGE=ON \
    -D CPACK_BINARY_DEB=ON \
    -D CPACK_BINARY_STGZ=OFF \
    -D CPACK_BINARY_TGZ=OFF \
    -D CPACK_BINARY_TZ=OFF \
    /opencv/opencv

cmake --build . --parallel "${OPENCV_BUILD_JOBS:-4}"
cmake --install .
cpack -G DEB
tar -czf "$OUTPUT_FILE" OpenCV*.deb
