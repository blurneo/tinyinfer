# Commands to build on macOS

SCRIPT_ROOT="$( cd "$(dirname "$0")" ; pwd -P)"

if [ -z ${IOS_PLATFORM+x} ]; then
    MAC_PLATFORM="MAC"
fi

if [ -z ${IOS_ARCH+x} ]; then
    MAC_ARCH=""
fi

if [ -z ${CMAKE_GENERATOR+x} ]; then
    CMAKE_GENERATOR="Xcode"
fi

if [ -z ${BUILD_DIR+x} ]; then
    BUILD_DIR=build-macos
fi

if [ -z ${BUILD_AND_INSTALL+x} ]; then
    BUILD_AND_INSTALL=ON
fi

if [ -z ${TI_BUILD_TYPE+x} ]; then
    TI_BUILD_TYPE=Debug
fi

if [ -z ${TI_ENABLE_SANITIZER+x} ]; then
    TI_ENABLE_SANITIZER=OFF
fi

if [ -z ${TI_Install_PREFIX+x} ]; then
    TI_Install_PREFIX="macOS"
fi

if [ -z ${INSTALL_DIR+x} ]; then
    INSTALL_DIR="release"
fi


###########################################
mkdir -p $SCRIPT_ROOT/$BUILD_DIR
cd $SCRIPT_ROOT/$BUILD_DIR

# Generate projects
echo "Generating TI projects"
cmake $SCRIPT_ROOT/.. \
    -G"$CMAKE_GENERATOR" \
    -DCMAKE_TOOLCHAIN_FILE=$SCRIPT_ROOT/../cmake/ios.toolchain.cmake \
    -DCMAKE_BUILD_TYPE=$TI_BUILD_TYPE \
    -DCMAKE_CONFIGURATION_TYPES=$TI_BUILD_TYPE \
    -DTI_Install_PREFIX="$TI_Install_PREFIX" \
    -DPLATFORM=$MAC_PLATFORM \
    -DARCHS="$MAC_ARCH" \
    -DINSTALL_DIR=$INSTALL_DIR \
    -DTI_ENABLE_SANITIZER=$TI_ENABLE_SANITIZER \
    || exit 1

echo "TI projects generate successfully"

if [ ${BUILD_AND_INSTALL} = "ON" ]; then
    echo "Building TI"
    cmake --build . --config $TI_BUILD_TYPE -- $@ || exit 1
    echo "TI build successfully"
fi
