cmake_minimum_required(VERSION 3.10)
project(HailoYOLO26_Pi5_Pipeline CXX)

# HailoRT and modern tracking libraries require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# ==========================================
# 1. Find System Dependencies
# ==========================================

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find HailoRT (Installed via the official .deb packages)
find_library(HAILORT_LIB NAMES hailort REQUIRED)

# Find Roboflow Trackers 
# Searching for the specific 'roboflow_trackers' library installed on the system
find_library(TRACKERS_LIB NAMES roboflow_trackers REQUIRED)

# ==========================================
# 2. Configure Include Directories
# ==========================================

# Point to the cpp/include folder from the root
include_directories(
    cpp/include
    ${OpenCV_INCLUDE_DIRS}
)

# ==========================================
# 3. Build Executables
# ==========================================

# -- A. Single Image Detection --
add_executable(detect_image cpp/src/detect_image.cpp)
target_link_libraries(detect_image 
    ${OpenCV_LIBS} 
    ${HAILORT_LIB}
)

# -- B. Real-Time Video Detection --
add_executable(detect_video cpp/src/detect_video.cpp)
target_link_libraries(detect_video 
    ${OpenCV_LIBS} 
    ${HAILORT_LIB}
)

# -- C. Real-Time Video Tracking (BoTSORT) --
add_executable(track_video cpp/src/track_video.cpp)
target_link_libraries(track_video 
    ${OpenCV_LIBS} 
    ${HAILORT_LIB}
    ${TRACKERS_LIB}  # External roboflow_trackers library linked here
)