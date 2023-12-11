# MSci-Project---Oscillating-Cosmological-Perturbations

This project contains Python scripts for solving differential equations using Magnus expansion, Spectral and Runge-Kutta type adaptive numerical integration algorithms.


## Abstract:
This project investigates methods to integrate coupled systems of first-order ordinary differential equations with highly oscillatory solutions. These results inform the adaptability of community codes used to solve the Einstein-Boltzmann equations in cosmological perturbation theory. We first clarify the distinctions between Magnus expansion and multiple-scale analysis based approaches and contrast their suitability for use in cosmological perturbation theory. We then motivate the improved empirical performance demonstrated by a proposed modified Magnus expansion method known as the “Jordan-Magnus” method. Following this we generalize this approach, implementing a numerical routine that can be applied to arbitrarily sized systems and find it doesn’t outperform standard Magnus expansion methods. We also develop and implement a Chebyshev spectral collocation stepping method which offers improved accuracy and run-time for a five variable toy cosmology compared to Runge-Kutta and Magnus methods as well as offering improved flexibility through adaptive order.



## Contents

The project folder contains the following files:

- `Master's Thesis Sankalan.pdf`: This is the project report.
- `matrix_decomposition_experiments.py`: This is a Jupyter notebook containing rough experiments for algorithms for approximate or iterative numerical diagonalisation and other similarity transformations.
- `cosmological_systems_code.py`: This script sets up equations and matrices for a specific cosmological system and solves them using the different integration methods.
- `Two and Three-Dimensional Systems Code.py`: This script sets up equations and matrices for simple airy/burst and other analytically solvable oscillatory ODEs than can be formulated in a 2d or 3d system.
- `requirements.txt`: This file lists the Python libraries that the scripts depend on.
- `README.md`: This file provides an overview of the project and instructions for setting up and running the scripts.

## Dependencies

The scripts use the following Python libraries:

- numpy
- sympy
- scipy
- matplotlib
- time
- sys

## Setup

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required Python libraries using pip:

```bash
pip install -r requirements.txt
