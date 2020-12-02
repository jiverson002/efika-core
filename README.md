The Efika project is a collection of utilities for computing with sparse
matrices. The project is broken up into several components, maintained in
separate Git repositories, which can be selectively included into a project.
Currently, the project contains the following *core components*:
* data structures and manipulation routines for sparse matrices, used by all
  other components in the project
  ([efika-core](https://github.com/jiverson002/efika-core)),
* I/O routines for reading and writing sparse matrix files
  ([efika-io](https://github.com/jiverson002/efika-io)), and
* implementations of well-known algorithms for calculating fixed-radius nearest
  neighbors ([efika-apss](https://github.com/jiverson002/efika-apss)).

Together these core components compose `libEfika` which can be built as a
standalone C library and linked against. To build the complete library, there
exists a *build component*
([efika-dist](https://github.com/jiverson002/efika-dist)) that collects all core
components into a single CMake project and builds it into a single library. In
addition, this build component includes the logic necessary to prepare the
library as a [Conan](https://docs.conan.io/en/latest/) package.

Apart from the components listed above, the project also contains the following
*auxiliary components*:
* a benchmark framework to measure the performance of competing implementations
  of data structures and algorithms within the core components
  ([efika-perf](https://github.com/jiverson002/efika-perf)), and
* a web app to visualize metrics produced by said benchmark framework in order
  to quickly identify performance regressions
  ([efika-stat](https://github.com/jiverson002/efika-stat)).

#### Rationale
The rationale for the above design is simple. Each of the components can and
should be developed independently of the others. This encourages me as a
developer to create clear and clean interfaces between the components and helps
me to debug logic and performance problems in the code. Keeping them in separate
Git repositories also allows their histories to remain clean and focused.
Finally, it means that as the project grows and other components are added,
users of the components will not need to build and link against all of them;
rather, components can be selectively included.
