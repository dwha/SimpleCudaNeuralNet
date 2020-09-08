# author: dwookha@gmail.com

cmake_minimum_required (VERSION 3.0)

project (SimpleCudaNeuralNet LANGUAGES CXX CUDA)

#set( CMAKE_BUILD_TYPE DEBUG )
set( CMAKE_BUILD_TYPE RELEASE )

find_package(CUDA REQUIRED)

# -g: for breakpoint debugging in vscode
set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wall --std=c++11 -g" )
set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG" )
set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -O3" )

message (STATUS "#############################################")
message (STATUS "SYSTEM_PROCESSOR: " ${CMAKE_SYSTEM_PROCESSOR})
message (STATUS "SYSTEM_NAME: " ${CMAKE_SYSTEM_NAME})
message (STATUS "CXX_COMPILER_ID: " ${CMAKE_CXX_COMPILER_ID})
message (STATUS "CXX_COMPILER: " ${CMAKE_CXX_COMPILER})
message (STATUS "CUDA_COMPILER: " ${CMAKE_CUDA_COMPILER})
message (STATUS "BUILD_TYPE: " ${CMAKE_BUILD_TYPE})
message (STATUS "C_FLAGS: " ${CMAKE_C_FLAGS})
message (STATUS "CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
message (STATUS "CXX_FLAGS_DEBUG: " ${CMAKE_CXX_FLAGS_DEBUG})
message (STATUS "CXX_FLAGS_RELEASE: " ${CMAKE_CXX_FLAGS_RELEASE})
message (STATUS "#############################################")

include_directories (
	${CMAKE_CURRENT_SOURCE_DIR}
)

file (GLOB HEADER_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "*.h"
)
file (GLOB CPP_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "*.cpp"
)

list (APPEND CUDA_NN_SRC ${HEADER_FILES} ${CPP_FILES} ${CU_FILES})

set_source_files_properties( ffCudaNn.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )

cuda_add_executable(
    ffCudaNn
    ${CUDA_NN_SRC}
)
