
cmake_minimum_required(VERSION 3.10)

project(Main)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDA REQUIRED)

include_directories(
    ./include/client/third/miracl
    ./include/client/third/cryptopp565
    ./include/client
    ./include/sever
    ./include/common
    ./include
)

# Find OpenCV package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find Blind library
find_library(BLIND_LIB libBlind-16.so PATHS /home/blind/extend/dym/Blind_Cmake)

# Check if library is found
if(NOT BLIND_LIB)
    message(FATAL_ERROR "Blind library not found")
endif()

message("Blind library found: ${BLIND_LIB}")



# 收集 src/client 下的所有 CUDA 源文件
file(GLOB_RECURSE CUDA_FILES "src/*.cu")

# 添加主程序源文件
# set(CPP_FILES "src/test.cpp")

# 将 CUDA 和 C++ 文件一起添加到可执行文件
cuda_add_executable(main ${CUDA_FILES})


# 娣诲姞渚濊禆鐨勫簱鏂囦欢
target_link_libraries(main  ${CMAKE_SOURCE_DIR}/lib/libcryptopp.a)
target_link_libraries(main  ${CMAKE_SOURCE_DIR}/lib/miracl.a)
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main ${BLIND_LIB})


set(CMAKE_CUDA_FLAGS "-arch=sm_30")

# cmake_minimum_required(VERSION 3.10)
# project(HEKANLib LANGUAGES CXX CUDA)

# set(CMAKE_CXX_STANDARD 14)
# enable_language(CUDA)

# # 包含头文件
# include_directories(
#     ./include/client/third/miracl
#     ./include/client/third/cryptopp565
#     ./include/client
#     ./include/sever
#     ./include/common
#     ./include
# )


# # 添加源文件
# file(GLOB SOURCES "src/*.cu")

# # 生成静态库
# add_library(hekan_static STATIC ${SOURCES})

# # 生成动态库（可选）
# add_library(hekan_shared SHARED ${SOURCES})

# # 可根据需要设置 CUDA 编译选项
# set_target_properties(hekan_static PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(hekan_shared PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
