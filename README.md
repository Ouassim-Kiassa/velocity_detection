# Dev Container Command Line Tools

This document provides an overview of the command line tools available in the dev container environment, along with their usage instructions.

## Tools

### 1. libGL.so.1

- **Command:**
- **Usage:**
  - Run script:  
    ```bash
       sudo apt-get update && sudo apt-get install -y libgl1
    ```

### 2. libxcb for OpenCV's GUI

- **Command:**
- **Usage:**
  - Run script:  
    ```bash
    sudo apt-get install -y libxcb-xinerama0 libxkbcommon-x11-0 libxcb1 libx11-6
    ```