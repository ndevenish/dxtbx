# Distributes under BSD licence

#.rst:
# FindCCTBX
# ---------------
# Find an existing CCTBX distribution
#
#
# ::
#   find_libtbx_module(<name> [REQUIRED])
#
# Search the tbx-repositories for a specific named module. If the
# ``REQUIRED`` parameter is specified, then an error is thrown if
# the module can not be found.
#
# ::
#   add_libtbx_module(<name> [INTERFACE] source1 [source2 ...])
#
# Register the current directory as a libtbx module, and read the module
# folder and sources to determine any hard/soft dependencies for the
# module. Then, generate dispatchers for any command_line scripts.
#
# A target with the same name as the module will be created. If the
# ``INTERFACE`` option is specified then this will be an interface
# target, otherwise it will be a library of the default library type (as
# specified by :variable:`BUILD_SHARED_LIBS`).
#
# If the target module is not an interface module, then sources must
# be present.

# Temporarily remove:
#                            [[GENERATED_FILES files...] | NO_REFRESH]
# If there is a ``libtbx_refresh.py`` file that generates source files for
# inclusion as part of the build, then the output files must be specified
# as a ``GENERATED_FILES`` argument. These files are passed to the
# ``add_libtbx_refresh_command` function. If there happens to be a
# ``libtbx_refresh.py`` script but it does not generate files, then you can
# pass the ``NO_REFRESH`` option to suppress the warning.

# Needs components - cctbx, scitbx


# Let's first try to find the libtbx environment location.

if (NOT TARGET Python::Interpreter)
    find_package(Python COMPONENTS Interpreter REQUIRED)
endif()

# Find the location of the libtbx build directory - where libtbx_env is
function(_cctbx_determine_libtbx_build_dir)
    # Try and read it from the environment
    message(DEBUG "Looking for libtbx build dir via environment LIBTBX_BUILD")
    if (DEFINED ENV{LIBTBX_BUILD})
        get_filename_component(_ABS_BUILD "$ENV{LIBTBX_BUILD}" ABSOLUTE BASE_DIR "${CMAKE_BINARY_DIR}")
        set(LIBTBX_ENV "${_ABS_BUILD}/libtbx_env")
        message(DEBUG "Checking ${LIBTBX_ENV}")
        if (EXISTS "${LIBTBX_ENV}")
            set(CCTBX_BUILD_DIR "${_ABS_BUILD}" CACHE FILEPATH "Location of CCTBX build directory")
            message(DEBUG "Found libtbx via environment LIBTBX_BUILD: ${_ABS_BUILD}")
            return()
        endif()
    endif()

    message(DEBUG "Looking for libtbx build dir via importing libtbx in python")
    execute_process(COMMAND ${Python_EXECUTABLE} -c "import libtbx.load_env; print(abs(libtbx.env.build_path))"
                    RESULT_VARIABLE _LOAD_ENV_RESULT
                    OUTPUT_VARIABLE _LOAD_LIBTBX_BUILD_DIR
                    OUTPUT_STRIP_TRAILING_WHITESPACE)

    if (NOT ${_LOAD_ENV_RESULT})
        # We found it via python import
        message(DEBUG "Got libtbx build path: ${CCTBX_BUILD_DIR}")
        set(CCTBX_BUILD_DIR "${_LOAD_LIBTBX_BUILD_DIR}" CACHE FILEPATH "Location of CCTBX build directory")
        return()
    endif()

endfunction()

# Read details for a single module out of libtbx_env and other info
function(_cctbx_read_module MODULE)
    cmake_path(SET _read_env_script NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/../read_env.py")
    execute_process(
        COMMAND ${Python_EXECUTABLE} "${_read_env_script}" "${CCTBX_BUILD_DIR}/libtbx_env"
        OUTPUT_VARIABLE _env_json
        RESULT_VARIABLE _result)
    if (_result)
        message(SEND_ERROR "Failed to read environment file: ${CCTBX_BUILD_DIR}/libtbx_env")
        return()
    endif()
    # We now have a json representation of libtbx_env - extract the entry for this modile
    string(JSON _module_json ERROR_VARIABLE _error GET "${_env_json}" module_dict ${MODULE})
    if (NOT _module_json)
        set("CCTBX_${MODULE}_FOUND" FALSE PARENT_SCOPE)
        return()
    else()
        set("CCTBX_${MODULE}_FOUND" TRUE PARENT_SCOPE)
    endif()
    # Now, ensure a target exists for this module
    set(_target "CCTBX::${MODULE}")
    if (NOT TARGET CCTBX::${MODULE})
        add_library(${_target} INTERFACE IMPORTED)
    endif()

    # Get the dist-wide include dir
    string(JSON _include_paths GET "${_env_json}" include_path)
    string(JSON _lib_paths GET "${_env_json}" include_path)

    # Work out what paths need to be included for this module
    string(JSON _n_dist_paths LENGTH "${_module_json}" dist_paths)
    math(EXPR _n_dist_paths "${_n_dist_paths} - 1")
    # We need to account for:
    # Algorithm: if folder has an include/ subdir:
    #               use that
    #            else:
    #               use the path above
    # - this accounts for most cases, and it seems unlikely that we will
    #   be importing uniquely from modules that this doesn't cover. There
    #   was an exhaustive mapping list in tbx2cmake but the new installed
    #   layout confuses that somewhat.
    foreach(_n RANGE "${_n_dist_paths}")
        string(JSON _dist_path GET "${_module_json}" dist_paths ${_n})
        if(NOT _dist_path)
            continue()
        endif()
        list(APPEND _dist_paths "${_dist_path}/include")
        if (EXISTS "${_dist_path}/include")
            list(APPEND _include_paths "${_dist_path}/include")
        else()
            cmake_path(GET _dist_path PARENT_PATH _result)
            list(APPEND _include_paths "${_result}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _include_paths)
    list(REMOVE_DUPLICATES _dist_paths)
    message("${MODULE}: ${_include_paths}")

    target_include_directories(${_target} INTERFACE ${_include_paths})

    # Find if this module has a "base" library, and any sub-libraries
    cmake_path(SET _modules_libs_file NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/../module_libraries.json")
    file(READ "${_modules_libs_file}" _modules_libs)
    string(JSON _module_libs GET "${_modules_libs}" "${MODULE}")
    if (_module_libs)
        # We have actual libraries to import as imported libraries
        # iterate over every key: value in the object
        string(JSON _n_libs LENGTH "${_module_libs}")
        math(EXPR _n_libs "${_n_libs} - 1")
        foreach(_n RANGE ${_n_libs})
            string(JSON _name MEMBER "${_module_libs}" ${_n})
            string(JSON _libname GET "${_module_libs}" "${_name}")
            if(_name STREQUAL base)
                message(DEBUG "Module ${_target} has root library ${_libname}")
            else()
                message(DEBUG "Extra library ${_target}::${_name} = ${_libname}")
            endif()
        endforeach()
    endif()
endfunction()


if (NOT CCTBX_BUILD_DIR)
    _cctbx_determine_libtbx_build_dir()
endif()

if (CCTBX_BUILD_DIR)
    message(DEBUG "Using build dir ${CCTBX_BUILD_DIR}")

    foreach(_comp IN LISTS CCTBX_FIND_COMPONENTS)
        # Only try and find this component if we didn't already find it
        if(NOT TARGET CCTBX::${_comp})
            _cctbx_read_module(${_comp})
        else()
            # We already had a target for this module, so mark found
            set("CCTBX_${_comp}_FOUND" TRUE)
        endif()
    endforeach()
    unset(_comp)
endif()





include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CCTBX
    REQUIRED_VARS CCTBX_BUILD_DIR
    HANDLE_COMPONENTS
)

message(FATAL_ERROR "Stop")


# include(CMakeParseArguments)
# include(${CMAKE_CURRENT_LIST_DIR}/JsonParser.cmake)

# # Store this location so we know where to look for relative files
# set(__TBXDistribution_list_dir ${CMAKE_CURRENT_LIST_DIR})

# function(find_libtbx_module name)
#   message(WARNING "find_libtbx_module currently does nothing")
# endfunction()

# # Handle env generation - accumulate global lists of modules and paths
# define_property(GLOBAL PROPERTY TBX_MODULES
#     BRIEF_DOCS "List of all TBX modules configured"
#     FULL_DOCS "List of all TBX modules configured")
# define_property(GLOBAL PROPERTY TBX_MODULES_PATHS
#     BRIEF_DOCS "List of all TBX modules paths configured"
#     FULL_DOCS "List of all TBX modules paths configured")
# # set_property(GLOBAL PROPERTY TBX_MODULES "")
# # set_property(GLOBAL PROPERTY TBX_MODULES "")

# get_filename_component(LIBTBX_ENV_PY ${CMAKE_CURRENT_LIST_DIR}/../write_libtbx_env.py ABSOLUTE)

# function(write_libtbx_env)
#   # Set up a target to write the environment
#   # if (NOT TARGET Python::Python)
#   #     message(FATAL_ERROR "No python interpreter imported executable Python::Python")
#   # endif()
#   get_property(tbx_modules GLOBAL PROPERTY TBX_MODULES)
#   get_property(tbx_modules_paths GLOBAL PROPERTY TBX_MODULES_PATHS)

#   execute_process(
#     COMMAND "${Python_EXECUTABLE}" ${LIBTBX_ENV_PY} "${tbx_modules}" "${tbx_modules_paths}"
#   )

#   # add_custom_command(
#   #   COMMAND Python::Python ${LIBTBX_ENV_PY} 
#   #     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
#   #     OUTPUT ${CMAKE_BINARY_DIR}/libtbx_env

# # add_custom_command(
# #   COMMAND Python::Python ${LIBTBX_REFRESH_PY} --root=${CMAKE_SOURCE_DIR} --output=${CMAKE_BINARY_DIR} ${refresh_script}
# #   OUTPUT ${TBXR_OUTPUT} 
# #   ${TBXR_UNPARSED_ARGUMENTS})

# endfunction()

# function(write_setup)
#   set(dispatcher_template "${__TBXDistribution_list_dir}/../setup.py.template")
#   configure_file(${dispatcher_template} ${destination})
# endfunction()

# function(add_tbx_module name)
#   cmake_parse_arguments(TBX "INTERFACE;NO_REFRESH" "" "" ${ARGN}) # GENERATED_FILES
#   set(TBX_MODULE ${name} PARENT_SCOPE)
#   # Record these at global scope
#   set_property(GLOBAL APPEND PROPERTY TBX_MODULES ${name})
#   set_property(GLOBAL APPEND PROPERTY TBX_MODULES_PATHS "${CMAKE_CURRENT_SOURCE_DIR}")

#   if(TBX_INTERFACE)
#     add_library( ${name} INTERFACE )
#     set(module_type ", interface")
#   else()
#     add_library( ${name} ${TBX_UNPARSED_ARGUMENTS})
#     install(TARGETS ${name} LIBRARY)
#   endif()

#   # # Look for a libtbx libtbx_refresh
#   # if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/libtbx_refresh.py)
#   #   if (NOT TBX_GENERATED_FILES)
#   #     if (NOT TBX_NO_REFRESH)
#   #       message(WARNING "TBX Module ${name} has a libtbx_refresh.py, but no generated-file list has been passed. Unknown output.")
#   #     endif()
#   #   else()
#   #     # We've been given a list of files, and also have a generator. Register them.
#   #     add_libtbx_refresh_command(${CMAKE_CURRENT_SOURCE_DIR}/libtbx_refresh.py OUTPUT ${TBX_GENERATED_FILES})
#   #   endif()
#   # endif()

#   # Generate dispatchers for this module
#   _generate_libtbx_dispatchers(${name} ${CMAKE_CURRENT_SOURCE_DIR})

#   message(STATUS "Registered TBX module ${name} (${${name}_DISPATCHER_COUNT} dispatchers${module_type})")
# endfunction()

# # Read the libtbx_config file for a specific <entry> and read into a variable named <entry>.
# function(_read_libtbx_config file entry)
#   # Read and parse
#   file(READ ${file} libtbx_config_contents)
#   sbeParseJson(config libtbx_config)
#   # Loop over entries to build the list
#   foreach(var ${config.${entry}})
#     # message("${var} = ${${var}}")
#     list(APPEND ${entry} ${config.${entry}_${var}})
#   endforeach()
#   # Export the variable
#   set(${entry} "${${entry}}" PARENT_SCOPE)
# endfunction()

# # ::
# #   _get_libtbx_dispatcher_rename(<file> <default> <output>)
# #
# # Read a source file and return a list of names of dispatchers for it.
# # <file> is the name of the source file to read. <default> is the default
# # name if there is no override specifiers in the file. <output> is the
# # name of the output variable to put the list of dispatcher names into.
# #
# # An override specification is an entry in the file somewhere of the form::
# #
# #   # LIBTBX_SET_DISPATCHER_NAME [some-new-name]
# #
# # specifying what the script should be named as a dispatcher. There can be
# # multiple entries, in which case more than one dispatcher should be generated.
# function(_get_libtbx_dispatcher_rename file default output)
#   # The regex to extract the dispatcher name override
#   set(DISPATCHER_REGEX "[ \\t]*#[ \\t]?LIBTBX_SET_DISPATCHER_NAME[ \\t]+([A-Za-z0-9_.]+)")

#   # Find the matches in the file
#   file(READ ${file} CONT)
#   string(REGEX MATCHALL "${DISPATCHER_REGEX}" matches "${CONT}")

#   # This gives us the full text of every match - we only want the matched group
#   # so iterate over the list building a new one
#   set(dispatcher_names "")
#   foreach(match ${matches})
#     string (REGEX REPLACE "${DISPATCHER_REGEX}" "\\1" match "${match}")
#     # message("Match after replace: ${match}")
#     list(APPEND dispatcher_names "${match}")
#   endforeach()
#   # Use the default if we had none set
#   if (NOT dispatcher_names)
#     set(dispatcher_names "${default}")
#   endif()
#   set(${output} "${dispatcher_names}" PARENT_SCOPE)
# endfunction()

# # ::
# #   _write_dispatcher(<destination> <target>)
# #
# # Determines the type of dispatcher required to call <target> and writes
# # it out to <destination>. Useful extra variables for template file:
# #   DISPATCHER_TARGET   The target script for the dispatcher to run
# #   PYTHON_EXECUTABLE   The full path to the python interpreter
# function(_write_dispatcher destination DISPATCHER_TARGET)
#   # Template depends on the type of file...
#   if (target MATCHES "\\.py$")
#     set(dispatcher_template "${__TBXDistribution_list_dir}/../dispatcher.py.template")
#   elseif(target MATCHES "\\.sh$")
#     set(dispatcher_template "${__TBXDistribution_list_dir}/../dispatcher.sh.template")
#   else()
#     message(WARNING "Unknown dispatcher type for target ${DISPATCHER_TARGET}; ignoring")
#     return()
#   endif()

#   configure_file(${dispatcher_template} ${destination})

#   # configure_file(${CMAKE_CURRENT_LIST_DIR}/../type_id_eq_h.template
#   #              ${CMAKE_BINARY_DIR}/include/boost_adaptbx/type_id_eq.h)
# endfunction()

# # ::
# #   _generate_libtbx_dispatchers(<name> <path>)
# #
# # Read a libtbx module path and generate dispatchers by reading the
# # command_line folder and any additional folders listed in the
# # libtbx_config file's "extra_command_line_locations" entry.
# #
# # <name> is used to create the default dispatcher name - <name>.<target>
# #
# function(_generate_libtbx_dispatchers name path)
#   # Read the libtbx_config to see if there are any extra locations to search
#   set(dispatcher_locations command_line)
#   if (EXISTS ${path}/libtbx_config)
#     _read_libtbx_config(${path}/libtbx_config extra_command_line_locations)
#     list(APPEND dispatcher_locations ${extra_command_line_locations})
#   endif()

#   # Find potential targets in each of these locations
#   foreach(dir ${dispatcher_locations})
#     # Find targets in this folder, relative to the module root
#     file(GLOB matches LIST_DIRECTORIES false RELATIVE ${path} ${dir}/*.py ${dir}/*.sh)
#     # Exclude items from this list that don't match the criteria; A
#     # script is a potential dispatcher IF:
#     #   - Filename ends with .py or .sh
#     #   - NOT __init__.py
#     #   - NOT A hidden file e.g. starts with "."
#     foreach(file ${matches})
#       if (    ${file} MATCHES "^(.*/)?__init__\\.py$"   # __init__.py
#           OR  ${file} MATCHES "^(.*/)?\\." )            # hidden files
#         continue()
#       else()
#         list(APPEND dispatcher_targets ${file})
#       endif()
#     endforeach()
#   endforeach()
#   list(LENGTH dispatcher_targets dispatcher_count)
#   # Pass this count up to the parent function
#   set(${name}_DISPATCHER_COUNT ${dispatcher_count} PARENT_SCOPE)

#   # This is where we could filter the dispatcher list further, remove
#   # previously-created dispatchers if they no longer exist, things like that.
#   # However, since we want to keep this relatively simple for now, just
#   # regenerating the build folder seems like it'd be easy enough.

#   # Process every dispatcher target we collected
#   foreach(target ${dispatcher_targets})
#     get_filename_component(target_stripped_name "${target}" NAME)
#     # Get the filename without final extension
#     string(REGEX REPLACE "\\.[^.]*$" "" target_stripped_name "${target_stripped_name}")
#     # Work out if we had any "rename" directives inside the file
#     _get_libtbx_dispatcher_rename(${target} "${name}.${target_stripped_name}" dispatcher_names)
#     foreach(name ${dispatcher_names})
#       # Write this dispatcher
#       _write_dispatcher(${CMAKE_BINARY_DIR}/bin/${name} ${path}/${target})
#     endforeach()
#   endforeach()
# endfunction()

# # ::
# #   _write_program_dispatcher(<destination> <target>)
# #
# # Writes a dispatcher for a specific executable
# function(_write_program_dispatcher destination DISPATCHER_TARGET)
#   set(dispatcher_template "${__TBXDistribution_list_dir}/../dispatcher.sh.template")
#   configure_file(${dispatcher_template} ${destination})
# endfunction()

# _write_program_dispatcher(${CMAKE_BINARY_DIR}/bin/dials.python ${Python_EXECUTABLE})
# _write_program_dispatcher(${CMAKE_BINARY_DIR}/bin/libtbx.python ${Python_EXECUTABLE})
# _write_program_dispatcher(${CMAKE_BINARY_DIR}/bin/libtbx.pytest "${Python_EXECUTABLE} -mpytest")

