| Component      | Build status |
| :------------- | :----------: |
|  core          | ![core](https://github.com/jiverson002/efika-core/workflows/CMake/badge.svg) |
|  io            | ![io](https://github.com/jiverson002/efika-io/workflows/CMake/badge.svg) |
|  apss          | ![apss](https://github.com/jiverson002/efika-apss/workflows/CMake/badge.svg?branch=nova) |
|  dist          | ![dist](https://github.com/jiverson002/efika-dist/workflows/CMake/badge.svg) |

# Table of contents
* [Building with CMake](#building-with-cmake)
* [Incorporating into an existing CMake project](#incorporating-into-an-existing-cmake-project)
  * [Using `add_subdirectory()`](#using-add_subdirectory)
  * [Using `find_package()`](#using-find_package)
* [Using the library](#using-the-library)
* [Hacking](#hacking)

# Overview

The Efika project is a collection of utilities for computing with sparse
matrices. The project is broken up into several components, maintained in
separate Git repositories, which can be [selectively included into a project](#incorporating-into-an-existing-cmake-project).
Currently, the project contains the following *core components*:
* data structures and manipulation routines for sparse matrices, used by all
  other components in the project
  ([core](https://github.com/jiverson002/efika-core)),
* I/O routines for reading and writing sparse matrix files
  ([io](https://github.com/jiverson002/efika-io)), and
* implementations of well-known algorithms for calculating fixed-radius nearest
  neighbors ([apss](https://github.com/jiverson002/efika-apss)).

Together these core components compose `libefika` which can be built as a
standalone C library and linked against. To build the complete library, there
exists a *build component*
([dist](https://github.com/jiverson002/efika-dist)) that collects all core
components into a single CMake project and builds it into a single library. In
addition, this build component includes the logic necessary to prepare the
library as a [Conan](https://docs.conan.io/en/latest/) package.

Apart from the components listed above, the project also contains the following
*extra components*:
* a benchmark framework to measure the performance of competing implementations
  of data structures and algorithms within the core components
  ([perf](https://github.com/jiverson002/efika-perf)), and
* a web app to visualize metrics produced by said benchmark framework in order
  to quickly identify performance regressions
  ([stat](https://github.com/jiverson002/efika-stat)).

Each core component defines a [public](#public-api) and a [private](#hacking)
API and is built as a [CMake object
library](https://cmake.org/cmake/help/latest/command/add_library.html#object-libraries).
The reason that each component is built as an object library, as opposed to a
static or shared, is so that components retain the ability to resolve symbols in
each other's private API. If the components were built as shared libraries, then
if the default symbol visibility were "hidden", components would not be able to
resolve symbols in other components private API. Using object libraries allows
components to access the private API of other components, yet prevents consumers
of the pre-built `libefika` from doing so. Note, this does not prevent someone
who is including a component directly into their project from accessing the
private API. In this case, the project incorporating the component is acting no
different than some other project Efika core component. At this point, I have
not come across an elegant solution to this problem. So, for the time being,
those who are including components directly into their project should behave
themselves.

#### Rationale
The rationale for the above design is simple. Each of the components can and
should be developed independently of the others. This encourages me as a
developer to create clear and clean interfaces between the components and helps
me to debug logic and performance problems in the code. Keeping them in separate
Git repositories also allows their histories to remain clean and focused.
Finally, it means that as the project grows and other components are added,
users of the components will not need to build and link against all of them, or for users of pre-built `libefika`, incur the transitive dependencies of unused components; rather, components can be selectively included.

# Building with CMake
Since core components are built as object libraries, they are not meant to be
built and installed separately. Rather, when project Efika is to be installed,
the packager must select which components will be built into the library to be
installed. This is one of the primary purposes of the build component, `dist`.
Among other things, it contains all of the logic necessary to select core
components to be built into a static or shared library and subsequently
installed to the system somewhere, including a CMake configuration file for
later [inclusion into existing CMake
projects](#incorporating-into-an-existing-cmake-project). To build target of the
build component is `libefika`. Building `libefika` follows the typical CMake
build procedure, i.e.:

```
git clone https://github.com/jiverson002/efika-dist.git
cd efika-dist
mkdir build
cd build
cmake ..
cmake --build .
```

Assuming that the build was successful and use of the *Makefile*
[generator](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html),
the library could then be installed with:     

```
make install
```

which may or may not require *sudo* privileges depending how the build was
[configured](https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html).

#### Selection of core components

To select core components which should be included in the build, there is a
CMake cache variable, `EFIKA_CORE_COMPONENTS`, which by default includes a list
of all known core components. By changing this list, the components included in
the build will be affected. For example, to build `libefika` and include only
the `apss` component, you could replace the CMake configure line above with:

```
cmake -DEFIKA_CORE_COMPONENTS="apss" ..
```

#### Automatic dependency fetching
It should be noted that in the above example, the `apss` component has a
dependency on `core`. In this case, the `core` component need not be included
explicitly, since it will be included automatically by
`[efika-apss](https://github.com/jiverson002/efika-apss)`, which will fetch
`[efika-core](https://github.com/jiverson002/efika-core)` and add it as a
subdirectory to the project. This is true for each of the Efika components,
namely, that they will fetch and make available any dependencies which they
have.       

# Incorporating into an existing CMake project
Internally, project Efika depends heavily on the CMake function
`FetchContent_MakeAvailable()` to retrieve core component dependencies and add
them as subdirectories to the current project.

## Using `add_subdirectory()`
There are two ways to include project Efika using the CMake `add_subdirectory()`
function. The first simply adds the build component as a subdirectory. The build
component in turn adds all known core components. If this is desired, it can be accomplished easily using the following:

```cmake
include(FetchContent)

FetchContent_Declare(dist
  GIT_REPOSITORY https://github.com/jiverson002/efika-dist.git)

FetchContent_MakeAvailable(dist)

target_link_libraries(<your_target>
  PRIVATE Efika::efika)
```

However, given that one of the goals of this project is to allow components to
be included individually, the more flexible way of including Efika core
components in a CMake project via `add_subdirectory()` is to fetch and make them
available separately, including only those components for which you have a need.
This could look like:

```cmake
include(FetchContent)

FetchContent_Declare(apss
  GIT_REPOSITORY https://github.com/jiverson002/efika-apss.git)

FetchContent_MakeAvailable(apss)

target_link_libraries(<your_target>
  PRIVATE Efika::apss)
```

See the [note above](#automatic-dependency-fetching) about dependency handling between Efika components.

## Using `find_package()`

The project Efika build component, `dist`, provides a CMake configuration
file so that after being built and installed, it can be easily found and
configured using the CMake function `find_package()`. Besides importing the
target `Efika::efika`, the configuration file also checks for the presence of
any requested components. This allows the consumer of the Efika library to check
that the install includes the components needed for their project. For example,
a project wanting to use the APSS functions distributed in the `apss` component,
could use the following:

```cmake
find_package(Efika CONFIG REQUIRED
  COMPONENTS apss)

target_link_libraries(<your_target>
  PRIVATE Efika::efika)
```

# Using the library
## Public API
Each component in the Efika project defines a public API that can be accessed
using its public header file, namely `efika/<component>.h`    

To control the visibility of symbols in each component, project Efika relies on
the export macros generated by the CMake function
[`generate_export_header()`](https://cmake.org/cmake/help/latest/module/GenerateExportHeader.html).
This CMake function creates a file, `efika/core/export.h`, which contains a set
of macros suitable for controlling the visibility of library symbols. All
symbols that are part of the public API of an Efika component are exported using
the macro `EFIKA_EXPORT`, like so:

``` c
EFIKA_EXPORT extern int EFIKA_public_api_global_variable;

EFIKA_EXPORT int EFIKA_public_api_function();
```

The macro `EFIKA_EXPORT` itself is part of the public API of the core component.
Although besides being available for inclusion in the `efika/core.h` header
file, it has no real utility to consumers of an Efika component. Depending on
the type of library being built, static or shared, and whether the current CMake
project is building the 'exported' symbols or consuming them, the core component
will also add the appropriate compiler definitions necessary to control the
visibility of said symbols. In particular, `EFIKA_STATIC_DEFINE` will be defined
whenever Efika components are being built into a static library,
`efika_core_EXPORTS` will be defined whenever Efika components are being built
into a shared library. Neither will be defined in the case that the Efika
components were built into a shared library which is now being linked against.
Together, this means that a consumer of an Efika component can safely enable
*\"hidden\"* visibility, thereby hiding any non-public symbols, without losing
the ability to resolve symbols from the public API. From a library maintainer's
perspective, this is an excellent way to prevent users of the library from
accessing internal symbols not intended for use outside of the library.   

To avoid namespace collisions, all public API symbols will be prefixed with
`EFIKA_`.

# Hacking
## Private API
`efika/<component>/<something>.h`

To avoid namespace collisions, all private API symbols will be prefixed with
`efika_`.

## Renaming macros

`efika/<component>/rename.h`
