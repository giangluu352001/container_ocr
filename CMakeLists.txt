cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(LVTN VERSION 0.1.0)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()    
  
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)

# Additional Include Directories
include_directories(${OpenCV_INCLUDE_DIRS} "${CMAKE_SOURCE_DIR}/")

AUX_SOURCE_DIRECTORY(./src SRCS)
add_executable(container-ocr ${SRCS})

# Additional Dependencies 
target_compile_features(container-ocr PUBLIC cxx_range_for)
target_link_libraries(container-ocr PUBLIC 
-Wl,--no-as-needed TorchVision::TorchVision -Wl,--as-needed 
"${TORCH_LIBRARIES}" ${OpenCV_LIBS})

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET container-ocr
                    POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${TORCH_DLLS}
                    $<TARGET_FILE_DIR:container-ocr>)
endif (MSVC)
   
message(STATUS "*** CMAKE ***: ${CMAKE_SOURCE_DIR}")
message(STATUS "*** TORCHVISION ***: ${TorchVision_DIR}")
message(STATUS "*** OPENCV ***: ${OpenCV_INCLUDE_DIRS}")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set_property(TARGET container-ocr PROPERTY CXX_STANDARD 17)
include(CPack)  