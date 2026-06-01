# 🏋️‍♂️ YOLO26 Training & ONNX Export

This directory contains everything needed to train a custom YOLO26 Nano model and export it to a Hailo-compatible ONNX format. 

Whether you want to train the model from scratch on your own dataset or just use our pre-trained weights to skip ahead to the compilation phase, you will find the resources here.

---

## 📂 Provided Assets

If you just want to test the inference pipeline without training a model yourself, we have provided our dataset and the final trained weights:

* **Dataset:** [Download the Bottle Detection Dataset (Roboflow)](https://universe.roboflow.com/mobilenetv2-izplp/bottle-dataset-nz9hb/dataset/5)
* **Trained PyTorch Model:** `model/bottle.pt` (The raw trained YOLO weights)
* **Exported ONNX Model:** `model/bottle.onnx` (Ready for Hailo compilation)

---

## 📓 The Training Notebook

The core of this directory is the `train_yolo_bottle.ipynb` Jupyter Notebook. It acts as an end-to-end tutorial covering:

1.  **Dynamic Dataset Configuration:** Automatically generates the `data.yaml` file required by Ultralytics, mapping your dataset paths and class names.
2.  **YOLO26 Nano Fine-Tuning:** Loads the base `yolo26n.pt` architecture and trains it on the custom dataset.
3.  **Visual Evaluation:** Plots the confusion matrix and runs sanity-check inferences on test images to verify bounding box accuracy.
4.  **Hardware-Specific Export:** Converts the PyTorch model into an ONNX graph formatted specifically for the Hailo Dataflow Compiler.

### Requirements
To run the notebook locally or in Google Colab, you need the following standard dependencies:
```bash
pip install ultralytics opencv-python matplotlib pyyaml
```
## ⚠️ CRITICAL: Hailo ONNX Export Rules

f you modify the notebook or train your own model, you must strictly adhere to these ONNX export parameters. The Hailo Dataflow Compiler (DFC) is highly specific about the graph structures it can parse.

In the notebook, the export command looks like this:
```python
best_model.export(
    format="onnx",
    imgsz=640,       
    opset=12,        
    simplify=True    
)
```
### Why these parameters?
* **`opset=12`**: The Hailo parser relies on highly stable, universally supported ONNX operators. Newer opsets may contain dynamic operations that cannot be physically mapped to the Hailo-8L NPU.
* **`simplify=True`**: This is mandatory. It removes redundant layers, constant-folds operations, and fuses Batch Normalization layers into Convolutional layers. Without this, the model will fail to compile.
* **`imgsz=640`**: Edge accelerators require static tensor shapes. Dynamic input sizes are not supported on the chip.

## ⏭️ Next Steps
Once you have your `bottle.onnx` file (either by training it yourself via the notebook or using the provided file in the `model/` directory), you are ready to compile it into a `.hef` file.

Head over to the `../export/` directory for the Docker compilation instructions!