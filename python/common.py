"""Common utilities for Hailo-8L inference on YOLO26 (NMS-free dual-head model)"""

import numpy as np
import cv2
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List
from hailo_platform import (
    VDevice, HEF, ConfigureParams, InputVStreamParams, 
    OutputVStreamParams, InferVStreams, HailoStreamInterface, FormatType
)

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Vectorized sigmoid"""
    return 1.0 / (1.0 + np.exp(-x))

# ============================================================================
# Common Image Operations
# ============================================================================

def letterbox_image(img, target_size=640, color=(114, 114, 114)):
    """Resize image with aspect ratio preservation (letterbox)"""
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h))
    
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    
    padded = np.full((target_size, target_size, 3), color, dtype=np.uint8)
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    
    return padded, scale, pad_w, pad_h

def load_and_preprocess_image(img_path: str, target_size: int = 640, normalize: bool = False) -> Tuple[np.ndarray, Tuple[int, int], float, int, int]:
    """Load and preprocess image for inference"""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    # YOLO models expect RGB, but cv2.imread loads as BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]
    
    padded, scale, pad_w, pad_h = letterbox_image(img, target_size)
    
    if normalize:
        padded = padded.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(padded, axis=0)
    else:
        input_tensor = np.expand_dims(padded, axis=0).astype(np.uint8)
    
    return input_tensor, (orig_h, orig_w), scale, pad_w, pad_h

def scale_detections_to_original(detections: List[dict], orig_h: int, orig_w: int, scale: float, pad_w: int, pad_h: int) -> List[dict]:
    """Scale detection coordinates from inference space to original image space"""
    for det in detections:
        det['x1'] = max(0, min((det['x1'] - pad_w) / scale, orig_w))
        det['y1'] = max(0, min((det['y1'] - pad_h) / scale, orig_h))
        det['x2'] = max(0, min((det['x2'] - pad_w) / scale, orig_w))
        det['y2'] = max(0, min((det['y2'] - pad_h) / scale, orig_h))
        
    return detections

@dataclass
class InferenceStats:
    """Runtime statistics for inference"""
    preprocess_time: float
    hailo_inference_time: float
    postprocess_time: float
    total_time: float
    hailo_output_shape: str
    final_output_shape: str


class DetectionPostProcessor:
    """Postprocess detections and draw bboxes using defined classes"""
    
    CLASSES = {0: 'bottle'} # Default fallback class
    
    @classmethod
    def set_classes(cls, class_dict: Dict[int, str]):
        """Allow users to inject custom classes dynamically"""
        cls.CLASSES = class_dict
        
    @classmethod
    def get_class_name(cls, class_id: int) -> str:
        return cls.CLASSES.get(class_id, f"Class_{class_id}")
    
    @staticmethod
    def draw_bboxes(image: np.ndarray, detections: list, thickness: int = 2) -> np.ndarray:
        """Draw bounding boxes on image"""
        img = image.copy()
        h, w = img.shape[:2]
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, det in enumerate(detections):
            x1, y1 = int(max(0, det['x1'])), int(max(0, det['y1']))
            x2, y2 = int(min(w, det['x2'])), int(min(h, det['y2']))
            
            color = colors[i % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            label = f"{det['cls_name']} {det['conf']:.2f}"
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return img


class HailoPythonInferenceEngine:
    """Encapsulates Hailo-8L backbone + Python head inference"""
    
    def __init__(self, hef_path: str, num_classes: int = 1):
        """Initialize the hybrid inference engine
        
        Args:
            hef_path: Path to the compiled .hef file
            num_classes: Number of classes the model was trained on. 
                         Critical for dynamic tensor shape mapping.
        """
        self.hef_path = hef_path
        self.num_classes = num_classes
        
        # Initialize Hailo
        self.target = VDevice()
        self.hef = HEF(hef_path)
        configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        
        # Setup vstreams
        self.input_vstream_params = InputVStreamParams.make(self.network_group)
        self.output_vstream_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        
        # Expected shape mapping dynamically built based on num_classes
        # (Batch, Height, Width, Channels)
        self.shape_to_name = {
            (1, 80, 80, self.num_classes): 'cls_80',
            (1, 40, 40, self.num_classes): 'cls_40',
            (1, 20, 20, self.num_classes): 'cls_20',
            (1, 80, 80, 4): 'reg_80',
            (1, 40, 40, 4): 'reg_40',
            (1, 20, 20, 4): 'reg_20',
        }

        print(f"✓ Hailo engine initialized: {hef_path}")
        print(f"✓ Python head configured for {self.num_classes} classes.")
    
    def _run_python_head(self, dequantized_results: Dict, conf_threshold: float) -> List[dict]:
        """Run python head logic on dequantized Hailo outputs (vectorized)."""
        
        tensors = {}
        found_shapes = []
        for _, data in dequantized_results.items():
            shape = data.shape
            found_shapes.append(shape)
            if shape in self.shape_to_name:
                name = self.shape_to_name[shape]
                tensors[name] = data
        
        required_tensors = ['cls_80', 'cls_40', 'cls_20', 'reg_80', 'reg_40', 'reg_20']
        missing = [t for t in required_tensors if t not in tensors]
        if missing:
            print(f"Error: Missing tensors from HEF: {missing}")
            print(f"Found shapes: {found_shapes}")
            print(f"Expected shapes: {list(self.shape_to_name.keys())}")
            return []

        STRIDES = [8, 16, 32]
        GRID_SIZES = [80, 40, 20]
        logit_threshold = -np.log(1.0 / conf_threshold - 1.0)
        
        results = []
        
        for scale_idx in range(len(STRIDES)):
            stride = STRIDES[scale_idx]
            grid_dim = GRID_SIZES[scale_idx]
            
            cls_data = tensors[f'cls_{grid_dim}'][0]  # (H, W, num_classes)
            reg_data = tensors[f'reg_{grid_dim}'][0]  # (H, W, 4)
            
            # Reshape to (H*W, C)
            cls_flat = cls_data.reshape(-1, self.num_classes)
            reg_flat = reg_data.reshape(-1, 4)
            
            # Vectorized: find max logit and class per anchor
            max_logits = cls_flat.max(axis=1)       
            class_ids = cls_flat.argmax(axis=1)      
            
            mask = max_logits > logit_threshold
            if not mask.any():
                continue
            
            indices = np.where(mask)[0]
            scores = sigmoid(max_logits[indices])
            cls = class_ids[indices]
            
            rows = indices // grid_dim
            cols = indices % grid_dim
            
            l = reg_flat[indices, 0]
            t = reg_flat[indices, 1]
            r = reg_flat[indices, 2]
            b = reg_flat[indices, 3]
            
            x1 = (cols + 0.5 - l) * stride
            y1 = (rows + 0.5 - t) * stride
            x2 = (cols + 0.5 + r) * stride
            y2 = (rows + 0.5 + b) * stride
            
            for j in range(len(indices)):
                cls_id_val = int(cls[j])
                results.append({
                    'x1': round(float(x1[j]), 2),
                    'y1': round(float(y1[j]), 2),
                    'x2': round(float(x2[j]), 2),
                    'y2': round(float(y2[j]), 2),
                    'conf': round(float(scores[j]), 4),
                    'cls_id': cls_id_val,
                    'cls_name': DetectionPostProcessor.get_class_name(cls_id_val)
                })
        
        return results
    
    def infer(self, input_data: np.ndarray, verbose: bool = False, save_output: bool = False, conf_threshold: float = 0.5) -> Tuple[List[dict], InferenceStats]:
        """Run hybrid inference pipeline with Python head"""
        stats = InferenceStats(
            preprocess_time=0, hailo_inference_time=0, 
            postprocess_time=0, total_time=0,
            hailo_output_shape="", final_output_shape=""
        )
        
        t_start = time.perf_counter()
        
        if verbose:
            print(f"[INFERENCE] Input shape: {input_data.shape}, dtype: {input_data.dtype}")
        
        with self.network_group.activate() as active_group:
            with InferVStreams(self.network_group, self.input_vstream_params, self.output_vstream_params) as infer_pipeline:
                
                # A. Hailo Backbone Inference
                t_hailo = time.perf_counter()
                hailo_results = infer_pipeline.infer(input_data)
                stats.hailo_inference_time = time.perf_counter() - t_hailo
                stats.hailo_output_shape = str({k: v.shape for k, v in hailo_results.items()})

                # B. Python Head
                t_post = time.perf_counter()
                detections = self._run_python_head(hailo_results, conf_threshold)
                stats.postprocess_time = time.perf_counter() - t_post
                stats.final_output_shape = f"{len(detections)} detections"

        stats.total_time = time.perf_counter() - t_start
        
        if verbose:
            print(f"[SUMMARY] Pipeline timing:")
            print(f"  Hailo:   {stats.hailo_inference_time*1000:7.2f}ms")
            print(f"  PyHead:  {stats.postprocess_time*1000:7.2f}ms")
            print(f"  Total:   {stats.total_time*1000:7.2f}ms")
        
        return detections, stats