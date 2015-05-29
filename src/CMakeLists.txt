cmake_minimum_required(VERSION 3.1)
project(fyp)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "C:/Users/Regina/Documents/FinalYearProject/C_stuff/bin")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -lgsl")

set(SOURCE_FILES main.c)
add_executable(fyp ${SOURCE_FILES} load_pgm.h load_pgm.c util.h util.c)