
add_library(monocular_calibration
	../include/MonocularCalibration.h MonocularCalibration.cpp
)
add_executable(calib main.cpp)
target_link_libraries(calib monocular_calibration ${OpenCV_LIBS})
#-----------------------------------------------------------------------------

add_executable(calib_new calib_new.cpp)
target_link_libraries(calib_new ${OpenCV_LIBS})

add_executable(out_calib out_calib.cpp)
target_link_libraries(out_calib ${OpenCV_LIBS})