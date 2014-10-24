project( ts )
find_package( OpenCV REQUIRED )
add_executable( ts ts )
target_link_libraries (ts ${OpenCV_LIBS})