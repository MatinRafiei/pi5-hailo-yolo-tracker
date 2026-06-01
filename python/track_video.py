import cv2
import time
import argparse
import numpy as np
import supervision as sv 
from trackers import BoTSORTTracker

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
    scale = min(target_size[0] / orig_h, target_size[1] / orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h = (target_size[0] - new_h) // 2
    pad_w = (target_size[1] - new_w) // 2
    canvas = np.full((target_size[0], target_size[1], 3), 114, dtype=np.uint8)
    canvas[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    if normalize:
        canvas = canvas.astype(np.float32) / 255.0
    input_data = np.expand_dims(canvas, axis=0)
    return input_data, (orig_h, orig_w), scale, pad_w, pad_h


def track_video_stream(args):
    """Run real-time object tracking on a camera stream using Hailo-8L and BoTSORT"""
    
    # Dynamically configure classes based on user input
    class_dict = {i: name for i, name in enumerate(args.labels)}
    DetectionPostProcessor.set_classes(class_dict)
    num_classes = len(args.labels)
    
    print(f"[Initializing Hailo Engine with {args.hef} for {num_classes} classes...]")
    engine = HailoPythonInferenceEngine(args.hef, num_classes=num_classes)
    print("✓ Engine loaded successfully.")

    # Dynamic Tracker & Annotator Configuration
    tracker = BoTSORTTracker(
        lost_track_buffer=args.track_buffer,
        minimum_consecutive_frames=args.min_frames,
        high_conf_det_threshold=args.high_conf_threshold, 
        track_activation_threshold=args.track_activation,
        minimum_iou_threshold_first_assoc=args.iou_first_assoc,
        minimum_iou_threshold_second_assoc=args.iou_second_assoc
    )
    
    box_annotator = sv.BoxAnnotator(thickness=args.box_thickness)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, smart_position=True)

    # Handle string camera indexes (like "0"), video paths, or IP URLs
    if args.camera_id.isdigit():
        cap = cv2.VideoCapture(int(args.camera_id))
    else:
        cap = cv2.VideoCapture(args.camera_id)
        
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera source: {args.camera_id}")
        return

    # Set camera capture resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.input_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.input_height)

    print(f"[Starting tracking stream ({args.input_width}x{args.input_height}). Press 'q' to quit.]")
    
    frame_count = 0
    start_time = time.time()
    target_size = (args.model_size, args.model_size)
    
    track_history = {}
    previous_frame_data = {}
    last_known_id = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame from camera.")
            break
            
        orig_h, orig_w = frame.shape[:2]
        
        # 1. Preprocess the frame
        input_data, orig_size, scale, pad_w, pad_h = preprocess_frame(
            frame, 
            target_size=target_size, 
            normalize=args.normalize
        )
        
        # 2. Run Inference
        t_infer_start = time.perf_counter()
        results, stats = engine.infer(input_data, verbose=False, save_output=False, conf_threshold=args.conf_threshold)
        infer_time = (time.perf_counter() - t_infer_start) * 1000
        
        # 3. Post-process boxes back to original coordinates
        results = scale_detections_to_original(results, orig_h, orig_w, scale, pad_w, pad_h)
        
        if args.debug:
            print(f"\n--- Frame {frame_count} ---")
            if len(results) == 0:
                print(" [i] OBJECT IS NOT FOUND")
        
        # 4. Extract arrays for Tracker mapping
        xyxy = []
        confidences = []
        class_ids = []
        tracking_xyxy = []
        
        for det in results:
            orig_box = [det['x1'], det['y1'], det['x2'], det['y2']]
            xyxy.append(orig_box) 
            
            # Inflate bounding box boundaries to optimize overlap calculations
            inflated_box = [
                orig_box[0] - args.pad_inflation,
                orig_box[1] - args.pad_inflation,
                orig_box[2] + args.pad_inflation,
                orig_box[3] + args.pad_inflation,
            ]
            tracking_xyxy.append(inflated_box)
            
            confidences.append(det['conf']) 
            class_ids.append(det['cls_id']) 
            
        # Diagnostic printing of raw inferences
        if args.debug:
            if not xyxy:
                print(" [i] Detector found nothing under current confidence thresholds.")
            else:
                for i, (box, conf) in enumerate(zip(xyxy, confidences)):
                    box_ints = [int(x) for x in box]
                    print(f" Raw Det {i}: Conf={conf:.2f}, Box={box_ints}")
            
        # Feed inputs to BoTSORT tracker
        if len(tracking_xyxy) > 0:
            tracker_detections = sv.Detections(
                xyxy=np.array(tracking_xyxy, dtype=np.float32),
                confidence=np.array(confidences, dtype=np.float32),
                class_id=np.array(class_ids, dtype=int)
            )
            tracked_detections = tracker.update(tracker_detections, frame)
        else:
            tracked_detections = tracker.update(sv.Detections.empty(), frame)
        
        if tracked_detections.tracker_id is not None:
            valid_mask = tracked_detections.tracker_id != -1
            tracked_detections = tracked_detections[valid_mask]
            
        # 5. Coordinate Transformations and Visualization Adjustments
        if tracked_detections.tracker_id is not None and len(tracked_detections.tracker_id) > 0:
            active_ids = set()
            draw_xyxy = tracked_detections.xyxy.copy()
            
            for i, tracker_id in enumerate(tracked_detections.tracker_id):
                # Reverse inflation step safely back to original bounds
                draw_xyxy[i][0] += args.pad_inflation
                draw_xyxy[i][1] += args.pad_inflation
                draw_xyxy[i][2] -= args.pad_inflation
                draw_xyxy[i][3] -= args.pad_inflation 
                
                draw_xyxy[i][0] = max(0, draw_xyxy[i][0])
                draw_xyxy[i][1] = max(0, draw_xyxy[i][1])
                draw_xyxy[i][2] = max(0, draw_xyxy[i][2])
                draw_xyxy[i][3] = max(0, draw_xyxy[i][3])                
                
                current_raw_box = draw_xyxy[i]
                current_conf = tracked_detections.confidence[i]
                active_ids.add(tracker_id)
                
                cx = (current_raw_box[0] + current_raw_box[2]) / 2
                cy = (current_raw_box[1] + current_raw_box[3]) / 2
                
                # Identity Swapping Quality Diagnostics
                if args.debug and last_known_id is not None and tracker_id != last_known_id:
                    print(f" [!!!] ID SWITCH DETECTED: {last_known_id} -> {tracker_id}")
                    if last_known_id in previous_frame_data:
                        prev_cx, prev_cy, prev_conf = previous_frame_data[last_known_id]
                        distance = ((cx - prev_cx)**2 + (cy - prev_cy)**2)**0.5
                        print(f"        Jump Distance: {distance:.1f} px")
                        print(f"        Prev Conf: {prev_conf:.2f} | Current Conf: {current_conf:.2f}")
                        
                previous_frame_data[tracker_id] = (cx, cy, current_conf)
                last_known_id = tracker_id
                
                # Track position temporal smoothing filter
                if tracker_id in track_history:
                    smoothed_box = (args.alpha * current_raw_box) + ((1 - args.alpha) * track_history[tracker_id])
                else:
                    smoothed_box = current_raw_box
                
                track_history[tracker_id] = smoothed_box
                tracked_detections.xyxy[i] = smoothed_box
                
            # Evict stale tracks
            keys_to_delete = [k for k in track_history.keys() if k not in active_ids]
            for k in keys_to_delete:
                del track_history[k]
    
            # Generate smart labels utilizing the dynamically assigned class names
            labels = [
                f"{DetectionPostProcessor.get_class_name(class_id)} #{tracker_id} ({confidence:.2f})"
                for tracker_id, confidence, class_id in zip(
                    tracked_detections.tracker_id, 
                    tracked_detections.confidence, 
                    tracked_detections.class_id
                )
            ]
            
            output_frame = box_annotator.annotate(scene=frame, detections=tracked_detections)
            output_frame = label_annotator.annotate(scene=output_frame, detections=tracked_detections, labels=labels)
        else:
            output_frame = frame
        
        # 6. Render telemetry overlays
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        
        cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(output_frame, f"Infer: {infer_time:.1f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # 7. Render Stream
        cv2.imshow("Hailo-8L Tracking + YOLO26", output_frame)
        # Break the loop if the user presses 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[Video stream closed.]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Object Tracking with Hailo-8L + BoTSORT")
    
    # Core IO Arguments
    parser.add_argument("--camera-id", type=str, default="0", 
                        help="Camera index or network/stream url (e.g. http://127.0.0.1:8080/video)")
    parser.add_argument("--hef", type=str, required=True, 
                        help="Path to compiled HEF model mapping matrix file")
    
    # Hardware & Model Tuning
    parser.add_argument("--labels", type=str, nargs='+', default=['bottle'],
                        help="List of class names the model was trained on, separated by spaces (e.g., --labels car person).")
    parser.add_argument("--input-width", type=int, default=640, help="Camera sensor capture width parameter")
    parser.add_argument("--input-height", type=int, default=480, help="Camera sensor capture height parameter")
    parser.add_argument("--model-size", type=int, default=640, help="Square dimensions required by model input layer")
    parser.add_argument("--normalize", action="store_true", help="Normalize input frames to [0,1]")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Detection threshold cutoff")
    
    # BoTSORT Hyperparameters
    parser.add_argument("--track-buffer", type=int, default=120, help="Frames to keep an inactive track in memory")
    parser.add_argument("--min-frames", type=int, default=1, help="Min frames needed to activate an object path")
    parser.add_argument("--high-conf-threshold", type=float, default=0.3, help="Confidence threshold for first association pass")
    parser.add_argument("--track-activation", type=float, default=0.85, help="Threshold to spin up brand new track path")
    parser.add_argument("--iou-first-assoc", type=float, default=0.1, help="IoU threshold ceiling for primary assignment")
    parser.add_argument("--iou-second-assoc", type=float, default=0.4, help="IoU threshold ceiling for secondary assignment")
    
    # Mathematical Smoothing & Preprocessing Hacks
    parser.add_argument("--pad-inflation", type=int, default=40, help="Pixel box inflation scale margin given to the tracker")
    parser.add_argument("--alpha", type=float, default=0.7, help="Smoothing exponential weight filter (0.0=Static, 1.0=Raw)")
    
    # Aesthetics & Diagnostics
    parser.add_argument("--box-thickness", type=int, default=2, help="Visual width weight of drawn box contours")
    parser.add_argument("--debug", action="store_true", help="Print advanced tracker jump diagnostics to stdout")
    
    args = parser.parse_args()
    track_video_stream(args)
