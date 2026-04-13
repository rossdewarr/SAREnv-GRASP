# GRASP for UAV SAR

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

- Forked SAREnv repository containing code and files for a 3rd year undergraduate research project investigating Greedy Randomised Adaptive Search Proceadures for UAV-based SAR.

# Algorithm Outline
- Greedy Randomised Construction
- 1-Opt Local Search
- Reactive Alpha Randomisation Parameter Selection

## Example Script

The `examples/` directory contains an example script which runs the GRASP alogrithm: `04_1_evaluate_GRASP_path.py`

# SAREnv

[![Version](https://img.shields.io/badge/version-1.0-green.svg)](https://github.com/namurproject/sarenv)
SAREnv is used here in an academic context as permitted under the MIT license. 

SAREnv is an open-access dataset and evaluation framework designed to support research in UAV-based search and rescue (SAR) algorithms. This toolkit addresses the critical need for standardized datasets and benchmarks in wilderness SAR operations, enabling systematic evaluation and comparison of algorithmic approaches including coverage path planning, probabilistic search, and information-theoretic exploration. 

# Dependencies
All dependencies remain the same as the original SAREnv release. Follow all install instructions from the original SAREnv README: https://github.com/namurproject/SAREnv

# IP Statement
The algorithm presented here is licensed under the MIT license which permits its use for both academic and commercial purposes.
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# Acknowlegments 
Thanks to all responsible for the creation of SAREnv and its provision as open-source software under highly permissible terms.

Thanks to Steve Bullock and Sid Reid for their supervision throughout the project.

# Contact
- Ross Dewar - The University of Bristol
- Email: ross.dewar.2023@bristol.ac.uk


### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/sarenv.git
cd sarenv

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Download Pre-generated Dataset

The repository includes pre-generated datasets stored using Git LFS (Large File Storage). To download the data files needed to run the examples:

```bash
# Install Git LFS if not already installed
# On Ubuntu/Debian:
sudo apt-get install git-lfs

# On macOS with Homebrew:
brew install git-lfs

# On Windows, download from: https://git-lfs.github.io/

# Initialize Git LFS in the repository
git lfs install

# Download the dataset files
git lfs pull
```
