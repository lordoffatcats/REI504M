#!/bin/bash

# Clear old loaded modules (I think?)
module purge

module load Stages/2026 # I think this just includes a bunch of software we might need. Perhaps this can be optimised to just importing what we need

# Required many GPU training (I don't know exactly what MPI is, got this from a template). 
module load GCC OpenMPI CUDA
module load MPI-settings/CUDA # (I think this is necessary too for multiple GPU's to work)

# Some base modules commonly used in AI
module load Python
module load matplotlib IPython git

# ML Frameworks
module load  PyTorch torchvision
# module load PyTorch-Lightning # This may be needed in the future

module load mpi4py # I don't know what this is exactly
