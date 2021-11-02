# include(TBXDistribution)

# add_tbx_module( dxtbx INTERFACE )
# target_include_directories( dxtbx INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/.. )

# add_libtbx_refresh_command( ${CMAKE_CURRENT_SOURCE_DIR}/libtbx_refresh.py )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
Python_add_library( dxtbx_ext
    boost_python/to_ewald_sphere_helpers.cc
    boost_python/ext.cpp
    boost_python/compression.cc )
target_link_libraries(dxtbx_ext PUBLIC CCTBX::scitbx)
# target_link_libraries( dxtbx_ext PUBLIC CCTBX::cctbx CCTBX::scitbx::boost_python TIFF::TIFF )

Python_add_library( dxtbx_format_nexus_ext format/boost_python/nexus_ext.cc )
target_link_libraries( dxtbx_format_nexus_ext PUBLIC hdf5::hdf5 CCTBX::scitbx )
# TIFF::TIFF CCTBX::cctbx CCTBX::scitbx::boost_python Boost::python 

Python_add_library( dxtbx_imageset_ext boost_python/imageset_ext.cc )
target_link_libraries( dxtbx_imageset_ext PUBLIC CCTBX::scitbx )
# hdf5::hdf5 TIFF::TIFF CCTBX::cctbx CCTBX::scitbx::boost_python Boost::python 

add_python_library( dxtbx_format_image_ext
    format/boost_python/image_ext.cc
    $<$<TARGET_EXISTS:CBFlib::cbf>:format/boost_python/cbf_read_buffer.cpp>
)
target_link_libraries( dxtbx_format_image_ext
    PUBLIC
    CCTBX::scitbx
    $<TARGET_NAME_IF_EXISTS:CBFlib::cbf>
)
# hdf5::hdf5 TIFF::TIFF CCTBX::cctbx CCTBX::scitbx::boost_python Boost::python 

add_python_library( dxtbx_model_ext
    model/boost_python/beam.cc
    model/boost_python/spectrum.cc
    model/boost_python/goniometer.cc
    model/boost_python/kappa_goniometer.cc
    model/boost_python/multi_axis_goniometer.cc
    model/boost_python/panel.cc
    model/boost_python/detector.cc
    model/boost_python/scan.cc
    model/boost_python/scan_helpers.cc
    model/boost_python/crystal.cc
    model/boost_python/parallax_correction.cc
    model/boost_python/pixel_to_millimeter.cc
    model/boost_python/experiment.cc
    model/boost_python/experiment_list.cc
    model/boost_python/model_ext.cc )
target_link_libraries( dxtbx_model_ext PUBLIC CCTBX::scitbx)
# CCTBX::cctbx CCTBX::scitbx::boost_python TIFF::TIFF Boost::python

add_python_library( dxtbx_masking_ext masking/boost_python/ext.cc )
target_link_libraries( dxtbx_masking_ext PUBLIC CCTBX::scitbx)
#  CCTBX::cctbx CCTBX::scitbx::boost_python Boost::python TIFF::TIFF

# find_package(pybind11 REQUIRED)

# pybind11_add_module(dxtbx_flumpy boost_python/flumpy.cc)
# target_link_libraries(dxtbx_flumpy PUBLIC Boost::python)
# target_compile_options(dxtbx_flumpy PRIVATE -fvisibility=hidden)
