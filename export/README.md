# 📦 Hailo-8L Model Compilation (ONNX to HEF)

This directory handles the most compute-intensive part of the pipeline: compiling the trained `.onnx` model into a highly optimized hardware binary (`.hef`) for the Hailo-8L M.2 AI HAT+.

Because the Hailo Dataflow Compiler (DFC) requires native x86_64 architecture and strict Python environments, we use Docker to create an isolated sandbox. This prevents dependency conflicts and ensures the compilation succeeds regardless of your host operating system.

---

## 📄 File Overview

To automate and secure this process, two utility files are included in this directory:

### 1. `Dockerfile`
A configuration file that builds a clean, isolated Ubuntu 22.04 sandbox equipped with Python 3.10 and necessary system C-headers. This ensures you do not pollute your host machine's Python environment with strict Hailo requirements.

### 2. `compile.sh`
An automated execution script that handles the two critical compilation steps:
1. **Calibration Duplication:** The Hailo quantizer strictly requires 1,024 images to perform INT8 calibration. If you have fewer images, this script automatically duplicates your existing images to safely bypass the compiler's limitation.
2. **Execution:** It triggers the `export.cli` toolkit targeting the `hailo8l` hardware architecture using your ONNX file.

---

## 🛠️ Step-by-Step Compilation Guide

### Step 1: Prepare the Workspace
Before starting, ensure you have placed the following items inside this `export/` directory:
1. Your exported model: `bottle.onnx`
2. A folder named `calib_images/` containing at least a few hundred sample images from your dataset (used for INT8 quantization).
3. **The Hailo Native Wheels:** Download these from the Hailo Developer Zone and place them in this folder:
   * `hailo_dataflow_compiler-x.xx.x-py3-none-any.whl`
   * `hailort_x.xx.x_amd64.deb`
   * `hailort-x.xx.x-cp310-cp310-linux_x86_64.whl`

### Step 2: Build the Docker Sandbox
Open your terminal in this directory and build the container image. This will download Ubuntu 22.04 and configure the Python workspace.

```bash
sudo docker build -t hailo-compiler .
```
### Step 3: Boot the Sandbox

Launch the container. The -v $(pwd):/workspace flag mounts your current directory inside the container, meaning any files generated inside the sandbox will be saved directly to your host machine.
```bash
sudo docker run -it -v $(pwd):/workspace hailo-compiler
```

### Step 4: Install Toolkit Dependencies (Inside Container)

Once inside the Docker terminal, run the following commands to install the required graph-visualization headers, the Hailo suite, and the custom YOLO26 compiler script:
```bash
# 1. Install System Headers
apt-get update && apt-get install -y graphviz graphviz-dev

# 2. Install Hailo Drivers and API bindings
apt-get install -y /workspace/hailort_*_amd64.deb
pip install /workspace/hailort-*-cp310-*.whl
pip install /workspace/hailo_dataflow_compiler-*.whl

# 3. Clone the YOLO26 Toolkit
git clone [https://github.com/DanielDubinsky/yolo26_hailo.git](https://github.com/DanielDubinsky/yolo26_hailo.git)
cd yolo26_hailo
pip install ultralytics
```
### Step 5: Run the Compilation Script

With the environment fully configured, run the automated compile script.

Note: The partition exploration phase is heavily CPU-bound as it searches for the optimal memory map for the Hailo-8L. It may take over an hour depending on your CPU.
```bash
# Make the script executable
chmod +x /workspace/compile.sh

# Run the compilation
/workspace/compile.sh
```
## Step 6: Deploy to Raspberry Pi

Once the script finishes, your new bottle.hef file will appear in your directory. You can now type exit to leave the Docker container.

Transfer the .hef file to your Raspberry Pi 5 to begin running live inferences!