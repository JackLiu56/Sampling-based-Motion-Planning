cmake_minimum_required(VERSION 2.8.3)
project(ackermann_client)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x")

###########
## Build ##
###########
find_package(catkin REQUIRED COMPONENTS roscpp ackermann_msgs cv_bridge)

catkin_package()
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)
include_directories(include ${catkin_INCLUDE_DIRS} 
			    ${Boost_INCLUDE_DIRS} )

add_executable(navigate navigate.cc gnn.cpp)
add_dependencies(navigate ${catkin_EXPORTED_TARGETS})
target_link_libraries(navigate ${catkin_LIBRARIES} ${OpenCV_LIBS})
