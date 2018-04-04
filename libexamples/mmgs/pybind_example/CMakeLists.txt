cmake_minimum_required(VERSION 3.0)
project(pymmgs)

find_package(PythonLibs 3.6 REQUIRED)
find_package(PythonInterp 3.6 REQUIRED)
set(PYBIND11_INCLUDE_DIR "$ENV{PYBIND11_INCLUDE_DIR}")
set(PYBIND11_PYTHON_VERSION 3.6)
#-DPYTHON_LIBRARY=~/anaconda/lib/libpython3.6m.dylib 
#-DPYTHON_INCLUDE_DIR=~/anaconda/include/python3.6m/
#set(pybind11_DIR "$ENV{HOME}/Workspace/libigl/external/pybind11/tools/")
add_subdirectory(${PYBIND11_INCLUDE_DIR}/../ pybind11)
#find_package(pybind11 REQUIRED)

pybind11_add_module(pymmgs MODULE py.cpp)
target_link_libraries(pymmgs PRIVATE pybind11::module)
target_link_libraries(pymmgs PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../../../build/lib/libmmgs.a")
set_target_properties(pymmgs PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                        SUFFIX "${PYTHON_MODULE_EXTENSION}")
target_include_directories(pymmgs PRIVATE "$ENV{EIGEN_INCLUDE_DIR}")
target_include_directories(pymmgs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../build/include/)


