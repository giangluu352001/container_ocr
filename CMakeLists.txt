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
           
list(APPEND CMAKE_FIND_ROOT_PATH "C:/C++Lib/")
set(CLIPPER2_INC
  "${CMAKE_FIND_ROOT_PATH}Clipper2Lib/include/clipper2/clipper.h"
  "${CMAKE_FIND_ROOT_PATH}Clipper2Lib/include/clipper2/clipper.core.h"
  "${CMAKE_FIND_ROOT_PATH}Clipper2Lib/include/clipper2/clipper.engine.h"
  "${CMAKE_FIND_ROOT_PATH}Clipper2Lib/include/clipper2/clipper.export.h"
  "${CMAKE_FIND_ROOT_PATH}Clipper2Lib/include/clipper2/clipper.minkowski.h"
  "${CMAKE_FIND_ROOT_PATH}Clipper2Lib/include/clipper2/clipper.offset.h"
  "${CMAKE_FIND_ROOT_PATH}Clipper2Lib/include/clipper2/clipper.rectclip.h"
)
    
set(CLIPPER2_SRC
  "${CMAKE_FIND_ROOT_PATH}Clipper2Lib/src/clipper.engine.cpp"
  "${CMAKE_FIND_ROOT_PATH}Clipper2Lib/src/clipper.offset.cpp"
  "${CMAKE_FIND_ROOT_PATH}Clipper2Lib/src/clipper.rectclip.cpp"
)  

# set (ONNXRUNTIME_ROOTDIR "${CMAKE_FIND_ROOT_PATH}onnxruntime-win-x64-1.14.1")
add_library(LVTN ${CLIPPER2_INC} ${CLIPPER2_SRC})
target_include_directories(LVTN PUBLIC "${CMAKE_FIND_ROOT_PATH}Clipper2Lib/include")
find_package( OpenCV REQUIRED )
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
     
  
AUX_SOURCE_DIRECTORY(./src SRCS)
add_executable(opencvtest ${SRCS})
# Additional Dependencies 
target_link_libraries( opencvtest PUBLIC TorchVision::TorchVision "${OpenCV_DIR}/opencv_world470.lib" "${TORCH_LIBRARIES}" LVTN)
 
# Additional Include Directories
include_directories( ${OpenCV_INCLUDE_DIRS} "${CMAKE_SOURCE_DIR}/")
# Additional Library Directories
link_directories( ${OpenCV_DIR})
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET opencvtest
                    POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${TORCH_DLLS}
                    $<TARGET_FILE_DIR:opencvtest>)
endif (MSVC)

message(STATUS "*** CMAKE ***: ${CMAKE_SOURCE_DIR}")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack) 