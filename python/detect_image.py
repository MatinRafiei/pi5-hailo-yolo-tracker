import argparse
import cv2
import time
import os
from common import (
    load_and_preprocess_image,
    scale_detections_to_original,
    DetectionPostProcessor,
    HailoPythonInferenceEngine
)

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO26 Hailo-8L Single Image Inference")
    parser.add_argument("image", type=str, help="Path to the input image")
    parser.add_argument("--hef", type=str, required=True, help="Path to the compiled .hef model")
    parser.add_argument("--labels", type=str, nargs="+", default=["bottle"],
                        help="List of class names (space separated). Example: --labels bottle cup")
    parser.add_argument("--model-size", type=int, default=640, help="Model input square size")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", type=str, default="output_detected.jpg", help="Output image path")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Input image '{args.image}' not found.")
        return

    # 1. Setup Classes Dynamically
    class_dict = {i: name for i, name in enumerate(args.labels)}
    DetectionPostProcessor.set_classes(class_dict)
   
    print(f"[Initializing Hailo Engine for {len(class_dict)} classes...]")

    # 2. Initialize Inference Engine
    # Pass the dynamic number of classes so common.py maps the tensors correctly
    engine = HailoPythonInferenceEngine(args.hef, num_classes=len(class_dict))

    # 3. Load and Preprocess Image
    print(f"[Loading image: {args.image}]")
    t_pre_start = time.perf_counter()
    input_tensor, (orig_h, orig_w), scale, pad_w, pad_h = load_and_preprocess_image(
        args.image,
        target_size=args.model_size,
        normalize=False # Hailo INT8 requires raw UINT8 [0-255] pixels
    )
    t_pre = time.perf_counter() - t_pre_start
    print(f"✓ Preprocessed to: {input_tensor.shape}")

    # 4. Run Inference
    print("[Running inference...]")
    raw_detections, stats = engine.infer(
        input_tensor,
        verbose=False,
        conf_threshold=args.conf_threshold
    )

    # 5. Scale Bounding Boxes back to the original image dimensions
    final_detections = scale_detections_to_original(
        raw_detections, orig_h, orig_w, scale, pad_w, pad_h
    )

    print(f"✓ Inference completed in {(stats.total_time)*1000:.2f}ms")
    print(f"✓ Found {len(final_detections)} detections above threshold {args.conf_threshold}")

    # 6. Render and Save
    orig_image = cv2.imread(args.image)
    for det in final_detections:
        print(f"  {det['cls_name']} {det['conf']:.2f} at [{det['x1']:.0f}, {det['y1']:.0f}, {det['x2']:.0f}, {det['y2']:.0f}]")

    annotated_image = DetectionPostProcessor.draw_bboxes(orig_image, final_detections)
    cv2.imwrite(args.output, annotated_image)
    print(f"✓ Output image saved to: {args.output}")

if __name__ == "__main__":
    main()
