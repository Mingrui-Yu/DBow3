# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mingrui/Mingrui/SLAMProject/DBoW/DBow3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mingrui/Mingrui/SLAMProject/DBoW/DBow3/build

# Include any dependencies generated for this target.
include mytest/CMakeFiles/test1.dir/depend.make

# Include the progress variables for this target.
include mytest/CMakeFiles/test1.dir/progress.make

# Include the compile flags for this target's objects.
include mytest/CMakeFiles/test1.dir/flags.make

mytest/CMakeFiles/test1.dir/test1.cpp.o: mytest/CMakeFiles/test1.dir/flags.make
mytest/CMakeFiles/test1.dir/test1.cpp.o: ../mytest/test1.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mingrui/Mingrui/SLAMProject/DBoW/DBow3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object mytest/CMakeFiles/test1.dir/test1.cpp.o"
	cd /home/mingrui/Mingrui/SLAMProject/DBoW/DBow3/build/mytest && /usr/bin/g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test1.dir/test1.cpp.o -c /home/mingrui/Mingrui/SLAMProject/DBoW/DBow3/mytest/test1.cpp

mytest/CMakeFiles/test1.dir/test1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test1.dir/test1.cpp.i"
	cd /home/mingrui/Mingrui/SLAMProject/DBoW/DBow3/build/mytest && /usr/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mingrui/Mingrui/SLAMProject/DBoW/DBow3/mytest/test1.cpp > CMakeFiles/test1.dir/test1.cpp.i

mytest/CMakeFiles/test1.dir/test1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test1.dir/test1.cpp.s"
	cd /home/mingrui/Mingrui/SLAMProject/DBoW/DBow3/build/mytest && /usr/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mingrui/Mingrui/SLAMProject/DBoW/DBow3/mytest/test1.cpp -o CMakeFiles/test1.dir/test1.cpp.s

mytest/CMakeFiles/test1.dir/test1.cpp.o.requires:

.PHONY : mytest/CMakeFiles/test1.dir/test1.cpp.o.requires

mytest/CMakeFiles/test1.dir/test1.cpp.o.provides: mytest/CMakeFiles/test1.dir/test1.cpp.o.requires
	$(MAKE) -f mytest/CMakeFiles/test1.dir/build.make mytest/CMakeFiles/test1.dir/test1.cpp.o.provides.build
.PHONY : mytest/CMakeFiles/test1.dir/test1.cpp.o.provides

mytest/CMakeFiles/test1.dir/test1.cpp.o.provides.build: mytest/CMakeFiles/test1.dir/test1.cpp.o


# Object files for target test1
test1_OBJECTS = \
"CMakeFiles/test1.dir/test1.cpp.o"

# External object files for target test1
test1_EXTERNAL_OBJECTS =

mytest/test1: mytest/CMakeFiles/test1.dir/test1.cpp.o
mytest/test1: mytest/CMakeFiles/test1.dir/build.make
mytest/test1: src/libDBoW3.so.0.0.1
mytest/test1: /usr/local/lib/libopencv_dnn.so.3.4.9
mytest/test1: /usr/local/lib/libopencv_highgui.so.3.4.9
mytest/test1: /usr/local/lib/libopencv_ml.so.3.4.9
mytest/test1: /usr/local/lib/libopencv_objdetect.so.3.4.9
mytest/test1: /usr/local/lib/libopencv_shape.so.3.4.9
mytest/test1: /usr/local/lib/libopencv_stitching.so.3.4.9
mytest/test1: /usr/local/lib/libopencv_superres.so.3.4.9
mytest/test1: /usr/local/lib/libopencv_videostab.so.3.4.9
mytest/test1: /usr/local/lib/libopencv_calib3d.so.3.4.9
mytest/test1: /usr/local/lib/libopencv_features2d.so.3.4.9
mytest/test1: /usr/local/lib/libopencv_flann.so.3.4.9
mytest/test1: /usr/local/lib/libopencv_photo.so.3.4.9
mytest/test1: /usr/local/lib/libopencv_video.so.3.4.9
mytest/test1: /usr/local/lib/libopencv_videoio.so.3.4.9
mytest/test1: /usr/local/lib/libopencv_imgcodecs.so.3.4.9
mytest/test1: /usr/local/lib/libopencv_imgproc.so.3.4.9
mytest/test1: /usr/local/lib/libopencv_core.so.3.4.9
mytest/test1: mytest/CMakeFiles/test1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/mingrui/Mingrui/SLAMProject/DBoW/DBow3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test1"
	cd /home/mingrui/Mingrui/SLAMProject/DBoW/DBow3/build/mytest && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
mytest/CMakeFiles/test1.dir/build: mytest/test1

.PHONY : mytest/CMakeFiles/test1.dir/build

mytest/CMakeFiles/test1.dir/requires: mytest/CMakeFiles/test1.dir/test1.cpp.o.requires

.PHONY : mytest/CMakeFiles/test1.dir/requires

mytest/CMakeFiles/test1.dir/clean:
	cd /home/mingrui/Mingrui/SLAMProject/DBoW/DBow3/build/mytest && $(CMAKE_COMMAND) -P CMakeFiles/test1.dir/cmake_clean.cmake
.PHONY : mytest/CMakeFiles/test1.dir/clean

mytest/CMakeFiles/test1.dir/depend:
	cd /home/mingrui/Mingrui/SLAMProject/DBoW/DBow3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mingrui/Mingrui/SLAMProject/DBoW/DBow3 /home/mingrui/Mingrui/SLAMProject/DBoW/DBow3/mytest /home/mingrui/Mingrui/SLAMProject/DBoW/DBow3/build /home/mingrui/Mingrui/SLAMProject/DBoW/DBow3/build/mytest /home/mingrui/Mingrui/SLAMProject/DBoW/DBow3/build/mytest/CMakeFiles/test1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : mytest/CMakeFiles/test1.dir/depend
