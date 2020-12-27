cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pretrained_classification)

find_package(Torch REQUIRED)
#find_package(TorchVision REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 ${TORCH_CXX_FLAGS}")

add_executable(classify src/main.cpp)
target_link_libraries(classify ${TORCH_LIBRARIES} ${OpenCV_LIBS})
