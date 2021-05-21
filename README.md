# Capstone

This project will focuses in the development of a system capable of recognizing
American Sign Language alphabet gestures and produce translations into text.
The system will use an Intel RealSense depth camera that uses coded light
technology, machine learning algorithms for classification, a database and a
graphical interface.

## Dependecies

### Linux

Visit the
[librealsense](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)
If using Ubuntu, follow the instructions for adding the repos and install
packages. Otherwise, following the instructions for building and installing the
libraries from source.

### Mac
Visit the
[librealsense](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)
 repository and clone it onto your local machine. Follow the instructions to build the packages from source, then copy the 
*.so and *.dylib files, 12 total, to this project's working directory (where the *.py files are). Follow the Python 
instructions to continue setting the project's dependancies.

### Windows
The librealsense driver should automatically be installed when plugged into your device. Install Python 3.710 on your device. Then Follow the Python 
instructions to continue setting the project's dependancies.

### Python

- Create a venv with `python3 -m venv venv`
- Activate the venv with `source venv/bin/activate`
- Ensure pip is at proper version with `pip install --upgrade pip`
- Install required pip packages with `python3 -m pip install -r requirements.txt`
- Run the program from the root directory using `python3 main.py`
