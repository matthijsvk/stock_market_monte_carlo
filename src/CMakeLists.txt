###############################################################################
# Source files
###############################################################################

set(SOURCES
        simulations.cpp
        helpers.cpp
        )

###############################################################################
# Linker / library
###############################################################################

add_library(${CMAKE_PROJECT_NAME} STATIC ${SOURCES})

# link built libraries
target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC
        fmt::fmt-header-only
        Threads::Threads
        )

<<<<<<< HEAD
# link with header-only libraries # TODO is this the right way?
=======
# link with header-only libraries # TODO is this the right way???
>>>>>>> f0acca86b047da70320906cad1055ac472314b06
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC "${csv_SOURCE_DIR}")