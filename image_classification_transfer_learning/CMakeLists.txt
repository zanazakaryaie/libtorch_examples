cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(Transfer_Learning)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 ${TORCH_CXX_FLAGS}")

include_directories(include)

add_executable(train "src/train.cpp" "src/utils.cpp" "src/RMFD.cpp")
add_executable(classify "src/classify.cpp" "src/utils.cpp")

target_link_libraries(train ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(classify ${TORCH_LIBRARIES} ${OpenCV_LIBS})
