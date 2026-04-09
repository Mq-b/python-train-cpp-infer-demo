cmake_minimum_required(VERSION 3.16)
project(YOLOClassifier VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(DEFINED ENV{VCPKG_LIB} AND NOT "$ENV{VCPKG_LIB}" STREQUAL "")
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{VCPKG_LIB}")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/bin")

find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets)
find_package(onnxruntime CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgcodecs imgproc)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/ImageUtils.h
    src/ImageUtils.cpp
    src/MainWindow.h
    src/MainWindow.cpp
    src/OnnxClassifier.h
    src/OnnxClassifier.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
    onnxruntime::onnxruntime
)

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION .
)

install(FILES "${CMAKE_SOURCE_DIR}/README.md"
    DESTINATION .
)

if(EXISTS "${CMAKE_SOURCE_DIR}/models/cat_vs_dog/best.onnx")
    install(FILES "${CMAKE_SOURCE_DIR}/models/cat_vs_dog/best.onnx"
        DESTINATION models/cat_vs_dog
    )
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/models/cat_vs_dog/labels.txt")
    install(FILES "${CMAKE_SOURCE_DIR}/models/cat_vs_dog/labels.txt"
        DESTINATION models/cat_vs_dog
    )
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/models/WellColumnClassification/best.onnx")
    install(FILES "${CMAKE_SOURCE_DIR}/models/WellColumnClassification/best.onnx"
        DESTINATION models/WellColumnClassification
    )
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/models/WellColumnClassification/labels.txt")
    install(FILES "${CMAKE_SOURCE_DIR}/models/WellColumnClassification/labels.txt"
        DESTINATION models/WellColumnClassification
    )
endif()
