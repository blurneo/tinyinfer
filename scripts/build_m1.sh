# Commands to build on macOS

MOTION_ROOT="$( cd "$(dirname "$0")" ; pwd -P)"

#sh $MOTION_ROOT/submodule.sh

if [ -z ${IOS_PLATFORM+x} ]; then
    MAC_PLATFORM="MAC"
fi

if [ -z ${IOS_ARCH+x} ]; then
    MAC_ARCH="arm64"
fi

if [ -z ${CMAKE_GENERATOR+x} ]; then
    CMAKE_GENERATOR="Xcode"
fi

if [ -z ${BUILD_DIR+x} ]; then
    BUILD_DIR=build-m1
fi

if [ -z ${BUILD_AND_INSTALL+x} ]; then
    BUILD_AND_INSTALL=ON
fi

if [ -z ${MOTION_BUILD_TYPE+x} ]; then
    MOTION_BUILD_TYPE=Release
fi

if [ -z ${MOTION_ENABLE_SANITIZER+x} ]; then
    MOTION_ENABLE_SANITIZER=OFF
fi

if [ -z ${MOTION_Install_PREFIX+x} ]; then
    MOTION_Install_PREFIX="macOS"
fi

if [ -z ${INSTALL_DIR+x} ]; then
    INSTALL_DIR="release"
fi

if [ -z ${MOTION_BUILD_TESTS+x} ]; then
    MOTION_BUILD_TESTS=ON
fi

if [ -z ${MOTION_BUILD_BENCH+x} ]; then
    MOTION_BUILD_BENCH=OFF
fi

if [ -z ${MOTION_BUILD_TOOLS+x} ]; then
    MOTION_BUILD_TOOLS=ON
fi

if [ -z ${MOTION_BUILD_ENGINE_TNN+x} ]; then
    MOTION_BUILD_ENGINE_TNN=ON
fi

###########################################
mkdir -p $MOTION_ROOT/$BUILD_DIR
cd $MOTION_ROOT/$BUILD_DIR

# Generate projects
echo "Generating MOTION projects"
cmake $MOTION_ROOT/.. \
    -G"$CMAKE_GENERATOR" \
    -DCMAKE_TOOLCHAIN_FILE=$MOTION_ROOT/../cmake/ios.toolchain.cmake \
    -DCMAKE_BUILD_TYPE=$MOTION_BUILD_TYPE \
    -DCMAKE_CONFIGURATION_TYPES=$MOTION_BUILD_TYPE \
    -DMOTION_Install_PREFIX="$MOTION_Install_PREFIX" \
    -DPLATFORM=$MAC_PLATFORM \
    -DARCHS="$MAC_ARCH" \
    -DENABLE_BITCODE=false \
    -DINSTALL_DIR=$INSTALL_DIR \
    -DMOTION_BUILD_TESTS=$MOTION_BUILD_TESTS \
    -DMOTION_BUILD_BENCH=$MOTION_BUILD_BENCH \
    -DMOTION_ENABLE_SANITIZER=$MOTION_ENABLE_SANITIZER \
    -DMOTION_BUILD_TOOLS=${MOTION_BUILD_TOOLS} \
    -DMOTION_BUILD_ENGINE_TNN=${MOTION_BUILD_ENGINE_TNN} \
    || exit 1

echo "MOTION projects generate successfully"

if [ ${BUILD_AND_INSTALL} = "ON" ]; then
    # NOTE: The "--config" argument matters when your build tool is a multi-
    #       configuration one, otherwise it is ignored
    #       E.g., for Xcode build tool, CMAKE_BUILD_TYPE will be ignored by
    #       cmake --build if "--config" is not specified

    echo "Building MOTION"
    cmake --build . --config $MOTION_BUILD_TYPE -- $@ || exit 1
    echo "MOTION build successfully"

    echo "Installing MOTION"
    cmake --build . --target install --config $MOTION_BUILD_TYPE || exit 1
    echo "MOTION install successfully"
fi
