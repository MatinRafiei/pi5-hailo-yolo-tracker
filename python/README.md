# Real-Time YOLO26 Inference & Tracking with Hailo-8L

This repository contains a high-performance Python inference and object tracking pipeline optimized for the **Hailo-8L AI accelerator**. It features a hybrid execution architecture: running the heavy feature extraction backbone directly on the Hailo hardware, while executing the NMS-free dual-head post-processing logic via a highly vectorized Python head.

---

## 🛠 Script Overview & Architectural Enhancements

### 1. `common.py`
* **Origin:** Adapted from the open-source repository [DanielDubinsky/yolo26_hailo](https://github.com/DanielDubinsky/yolo26_hailo).
* **Our Modification:** The original implementation was hardcoded strictly for the COCO dataset (80 classes) with fixed output tensor shape mappings. We re-engineered the script to **support any arbitrary number of custom classes and custom naming conventions dynamically**. By decoupling the tensor shape assertions from a hardcoded constant, the `HailoPythonInferenceEngine` and `DetectionPostProcessor` automatically adapt their network parsing layers based on the run-time `--labels` flag.

### 2. `detect_image.py`
* **Origin:** Adapted from the original repository's single-image implementation.
* **Our Modification:** Fully updated to align with our dynamic multi-class architecture. It now accepts the `--labels` parameter to configure the underlying custom Python post-processing head on the fly, alongside a tunable `--model-size` argument to handle models compiled at resolution footprints other than the default $640 \times 640$.

### 3. `detect_video.py`
* **Origin:** Custom-written script for real-time video stream processing.
* **Key Features:** Implements an efficient live loop around an OpenCV `VideoCapture` pipeline (supporting hardware cameras, local files, or IP streams). It includes low-overhead hardware frame pre-processing (letterboxing, color-space correction, and batch-dimension padding) and overlays live runtime performance metrics (FPS counter, raw Hailo execution time, and total detection tallies).

### 4. `track_video.py`
* **Origin:** Custom-written advanced multi-object tracking platform.
* **Key Features & Advanced Logic:**
  * **BoTSORT Tracker Integration:** Leverages the robust multi-object `BoTSORTTracker` via the `supervision` library to persist object identity paths across frames.
  * **Bounding Box Padding Inflation (`--pad-inflation`):** To counter edge-case frame noise where tight boundaries might slip past the tracker's primary association step, a temporary inflation margin is added to the bounding box inputs. This optimizes overlap calculations (`IoU`) during association and is perfectly reversed before rendering.
  * **Exponential Moving Average Smoothing (`--alpha`):** Implements a temporal smoothing filter on the bounding box coordinates across successive frames:
    $$\text{Box}_{\text{smooth}} = \alpha \cdot \text{Box}_{\text{current}} + (1 - \alpha) \cdot \text{Box}_{\text{previous}}$$
    This drastically minimizes visual jitter caused by minor variations in hardware inference outputs.
  * **Identity-Swapping Diagnostics (`--debug`):** When debug mode is active, the pipeline tracks spatial centroids across frames. If an ID switch occurs, it calculates the jump distance in pixels and outputs live diagnostic logs to `stdout` to evaluate tracking stability.

---

## 🚀 Usage Guide

Ensure your Hailo RT environment is active and required libraries (`opencv-python`, `supervision`, `hailo_platform`) are installed.

### 1. Single Image Inference
Run object detection on a static image file using a custom-trained model:
```bash
python detect_image.py /path/to/image.jpg \
    --hef /path/to/model.hef \
    --labels bottle cap label \
    --model-size 640 \
    --output output_result.jpg
```

### 2. Real-Time Camera Stream Detection
Run real-time video processing directly from a USB web camera (device index 0):
```python
python detect_video.py \
    --camera-id 0 \
    --hef /path/to/model.hef \
    --labels person drone car \
    --conf-threshold 0.30
```

### Advanced Live Object Tracking (BoTSORT)
Run the tracking pipeline with full temporal smoothing and inflation logic enabled:
```python
python track_video.py \
    --camera-id 0 \
    --hef /path/to/model.hef \
    --labels drone battery motor \
    --pad-inflation 40 \
    --alpha 0.7 \
    --track-buffer 120 \
    --debug
```

## ⚙️ Key Configuration Arguments

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--hef` | `str` | *Required* | Path to your compiled Hailo Executable Format (`.hef`) file. |
| `--labels` | `str` | `['bottle']` | Space-separated list of training classes matching your model's classification index. |
| `--model-size` | `int` | `640` | Square dimensions expected by your model input layer. |
| `--pad-inflation` | `int` | `40` | Pixel margin temporarily added to bounding boxes for association stability. |
| `--alpha` | `float` | `0.7` | EMA smoothing multiplier. Closer to `1.0` favors raw output; closer to `0.0` favors history. |
| `--track-buffer` | `int` | `120` | Total number of frames to preserve an inactive or hidden object path in memory. |
| `--debug` | `flag` | `False` | Enables printing of real-time coordinate logs and identity jump distances. |