import cv2
import time
import argparse
import numpy as np

# Import the necessary classes from your existing environment
from common import (
    HailoPythonInferenceEngine, 
    DetectionPostProcessor, 
    scale_detections_to_original
)

def preprocess_frame(frame, target_size=(640, 640), normalize=False):
    """
    Real-time preprocessing for a raw OpenCV frame.
    Handles letterbox resizing, BGR to RGB conversion, and batch dimension padding.
    """
    orig_h, orig_w = frame.shape[:2]
    
    # Calculate scaling factor to fit within target size while maintaining aspect ratio
    scale = min(target_size[0] / orig_h, target_size[1] / orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    
    # Resize image
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Calculate padding to reach target size
    pad_h = (target_size[0] - new_h) // 2
    pad_w = (target_size[1] - new_w) // 2
    
    # Create canvas and place resized image inside
    canvas = np.full((target_size[0], target_size[1], 3), 114, dtype=np.uint8)
    canvas[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    
    # CRITICAL: Swap BGR (OpenCV default) to RGB (YOLO training default)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    
    if normalize:
        canvas = canvas.astype(np.float32) / 255.0
        
    # Add batch dimension: (1, H, W, 3)
    input_data = np.expand_dims(canvas, axis=0)
    
    return input_data, (orig_h, orig_w), scale, pad_w, pad_h


def detect_video_stream(args):
    """Run real-time detection on a camera stream using Hailo-8L"""
    
    # Dynamically configure classes based on user input
    class_dict = {i: name for i, name in enumerate(args.labels)}
    DetectionPostProcessor.set_classes(class_dict)
    num_classes = len(args.labels)
    
    print(f"[Initializing Hailo Engine with {args.hef} for {num_classes} classes...]")
    engine = HailoPythonInferenceEngine(args.hef, num_classes=num_classes)
    print("✓ Engine loaded successfully.")

    # Open the video stream (handles device index or virtual video paths/IP cams)
    if args.camera_id.isdigit():
        cap = cv2.VideoCapture(int(args.camera_id))
    else:
        cap = cv2.VideoCapture(args.camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera source: {args.camera_id}.")
        return

    # Set camera capture resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.input_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.input_height)

    print(f"[Starting video stream ({args.input_width}x{args.input_height}). Press 'q' to quit.]")
    
    # FPS tracking variables
    frame_count = 0
    start_time = time.time()
    target_size = (args.model_size, args.model_size)
        
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame from camera.")
            break
            
        orig_h, orig_w = frame.shape[:2]
        
        # 1. Preprocess the raw frame
        input_data, orig_size, scale, pad_w, pad_h = preprocess_frame(
            frame, 
            target_size=target_size, 
            normalize=args.normalize
        )
        
        # 2. Run inference
        t_infer_start = time.perf_counter()
        results, stats = engine.infer(input_data, verbose=False, save_output=False, conf_threshold=args.conf_threshold)
        infer_time = (time.perf_counter() - t_infer_start) * 1000 # in ms
        
        # 3. Post-process and scale boxes back to the original video resolution
        results = scale_detections_to_original(results, orig_h, orig_w, scale, pad_w, pad_h)
        
        # 4. Draw bounding boxes onto the frame
        output_frame = DetectionPostProcessor.draw_bboxes(frame, results, thickness=args.box_thickness)
        
        # 5. Calculate and display FPS and Inference Time
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        
        # Overlay metrics on the top-left corner
        cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(output_frame, f"Infer: {infer_time:.1f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
        # 6. Display the live feed
        cv2.imshow("Hailo-8L Real-Time YOLO26", output_frame)
        
        # Break the loop if the user presses 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Clean up hardware resources when the loop exits
    cap.release()
    cv2.destroyAllWindows()
    print("[Video stream closed.]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time USB Camera Detection with Hailo-8L + Python Head")
    
    # Hardware & Model Configuration
    parser.add_argument("--camera-id", type=str, default="0", 
                        help="Camera ID index (e.g., 0) or video file path / IP stream URL")
    parser.add_argument("--hef", type=str, required=True, 
                        help="Path to the compiled Hailo Executable Format (.hef) model")
    
    # Model Input Tuning
    parser.add_argument("--labels", type=str, nargs='+', default=['bottle'],
                        help="List of class names the model was trained on, separated by spaces (e.g., --labels car person).")
    parser.add_argument("--model-size", type=int, default=640, 
                        help="The input square dimensions the YOLO model expects (e.g., 640)")
    parser.add_argument("--normalize", action="store_true", 
                        help="Normalize input pixels to [0,1] float32 (omit if the HEF model takes raw uint8)")
    parser.add_argument("--conf-threshold", type=float, default=0.25, 
                        help="Confidence threshold for bounding box filtering")
    
    # Camera Stream Resolution
    parser.add_argument("--input-width", type=int, default=640, 
                        help="Set frame width parameter for camera hardware capture")
    parser.add_argument("--input-height", type=int, default=480, 
                        help="Set frame height parameter for camera hardware capture")
    
    # Visualization Tuning
    parser.add_argument("--box-thickness", type=int, default=2, 
                        help="Line thickness for bounding boxes drawn over the frames")
    
    args = parser.parse_args()
    detect_video_stream(args)
