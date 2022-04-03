## Probabilistic Semantic Slam
This repository contains code for two separate tasks:

  The modelling / extraction of semantic assignment problems from the KITTI SLAM dataset

  The comparison / computation of probabilistic slam weights as described in the publication cited below
  
This is also, ideally, a time-saver for anyone trying to develop a semantic SLAM framework of their own, as the framework of semantic data extraction is a central part of a full SLAM framework. The assignment code is also well isolated, ideally making it simple to lift and place into other frameworks as desired. 

## Motivation
Semantic measurements are more human readable, unique/sparse in the environment, and higher information than pixel/geometric measurements. They are also more robust to lighting/viewpoint changes, a car is a car at day or night, from the left or the right. Probabilistic assignment allows for robust assignment in the face of uncertainty without throwing away any measurements, which is a requirement when there are only 3-5 measurements per frame! The only other work in the literature doing this used the matrix permanent, which is computationally infeasible for anything but tiny problems, and wasn't open source. I aimed to fix both of those problems.

## Video / Pictures

[![@youtube Probabilistic Assignment for Semantic SLAM](photoSamples/thumbnail.png)](https://youtu.be/-yuNgoN7JAI)

![Assignment Enumeration vs. Permanent Computation Time](/photoSamples/timeScatLabelled2.pdf "Assignment Enumeration vs. Permanent Computation Time")

![Assignment Enumeration vs. Permanent Accuracy](/photoSamples/errOrdStats.pdf "Assignment Enumeration vs. Permanent Accuracy")

## Installation & Prerequisites
This code is built primarily on [GTSAM](https://github.com/borglab/gtsam), [GTSAM-QUADRICS](https://github.com/best-of-acrv/gtsam-quadrics), Eigen and OpenCV(C++ and Python). For plotting, I have used [Matplot++](https://alandefreitas.github.io/matplotplusplus/). 

If you are interested in just the code which computes the probabilistic assignment weights (located in assignment.cpp, and shortestPath.cpp), then none of the previously cited prerequisites are needed, except for potentially Eigen (depending on usage).

## Compilation
I use CMake for compilation, and have included the cmake file which works with my installation of the prerequisites, but your installation of the required packages will almost surely change the format of the CMake file. 

## How to use?
With all prerequisites installed:

```sh
$ mkdir build
$ cd build
$ cmake ..
$ cmake --build .
```

Which should create 2 executables, semSlamRun (the semantic SLAM data extraction), and compMethods (for comparing the assignment and permanent methods). 

To use, run the following command (in the build directory, or if desired, run from above the build directory and add build/ to the executable path)
```sh
$ ./semslamRun
```
For an explanation of the run options, see the beginning of system.cpp

To run the assignment comparison code, in the same folder run
```sh
$ ./compMethods o30_p0_k200_perm0_net1
```
where "o30_p0_k200_perm0_net1" is the default ID string from the default semslamRun code.

## Publication
[Elad Michael, Tyler Summers, Tony A. Wood, Chris Manzie, and Iman Shames. "Probabilistic Data Association for Semantic SLAM at Scale." arXiv preprint arXiv:2202.12802 (2022).](https://arxiv.org/pdf/2202.12802.pdf)

#### Anything else that seems useful
I'm happy to answer questions, just reach out!
