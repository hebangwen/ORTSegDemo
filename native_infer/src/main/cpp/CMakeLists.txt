# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.18.1)

# Declares and names the project.

project("native_infer")

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
        native_infer

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        native_infer.cpp)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

# MARK: use `.so` library from prebuilt onnxruntime aar file (version 1.11.0)
set(ONNXRUNTIME_ROOTDIR "include/onnxruntime")
include_directories("${ONNXRUNTIME_ROOTDIR}/core/session")
add_library(onnxruntime SHARED IMPORTED)
set(ONNXRUNTIME_LINK_LIBRARY_PATH
        "${CMAKE_SOURCE_DIR}/libs/${CMAKE_ANDROID_ARCH_ABI}/libonnxruntime.so")
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${ONNXRUNTIME_LINK_LIBRARY_PATH})

if (NOT onnxruntime)
    message(WARNING "onnxruntime lib not found in ${ONNXRUNTIME_LINK_LIBRARY_PATH}")
endif()

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        native_infer

        # Links the target library to the log library
        # included in the NDK.
        ${log-lib}
        onnxruntime)