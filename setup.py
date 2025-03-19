import subprocess
from setuptools import setup, find_packages

# Function to install the submodule in editable mode
def install_nnUNet():
    # Path to the submodule
    submodule_path = 'src/models/nnUNet'
    
    # Run pip install -e inside the submodule
    subprocess.check_call(['pip', 'install', '-e', submodule_path])

# Run the installation of nnUNet before setup
install_nnUNet()

# Standard setup configuration
setup(
    name='oct_segmentation',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # Add your project dependencies here
    ],
    # Other setup arguments as needed
)
