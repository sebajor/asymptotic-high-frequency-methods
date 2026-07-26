cmake_minimum_required(VERSION 3.25)
project(regrid)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


#add_compile_options(-Wall -Wextra -Wpedantic)

include_directories(../includes
)

add_executable(${PROJECT_NAME} 
    regrid.cpp
    #../../tcp_server_multi_client/source/TcpServer.cpp
    #../../TcpUtils/source/TcpSocket.cpp
)

find_package(Threads REQUIRED)
target_link_libraries(regrid PRIVATE Threads::Threads)


#find_library(fft_lib fftw3)
#if(${fft_lib})
#    message("No library fftw3 found")
#else()
    #message("fftw3 lib found as ${fft_lib}$")
#endif()


#target_link_libraries(${PROJECT_NAME} ${fft_lib})
