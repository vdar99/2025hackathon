#!/usr/bin/env python3
"""
YOLOv8 Object Detection for OVR Live Video Feed
"""
import zmq
import base64
import time
import cv2
import numpy as np
import time
import argparse
import os
import json
from ultralytics import YOLO

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='YOLOv8 Object Detection for OVR Video Feed')
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                        help='YOLOv8 model path (default: yolov8n.pt)')
    parser.add_argument('--conf', type=float, default=0.25, 
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--device', type=int, default=0, 
                        help='Camera device ID (default: 0)')
    parser.add_argument('--save', action='store_true', 
                        help='Save output video')
    parser.add_argument('--output', type=str, default='output.mp4', 
                        help='Output video path (default: output.mp4)')
    parser.add_argument('--json-dir', type=str, default='detections',
                        help='Directory to save JSON detection files')
    parser.add_argument('--interval', type=float, default=3.0,
                        help='Time interval between JSON exports in seconds (default: 3.0)')
    
    # Parse arguments
    args = parser.parse_args()
    return args

def initialize_camera(device_id=0):
    """Initialize camera capture."""
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera device {device_id}")
    return cap

def initialize_model(model_path):
    """Initialize YOLOv8 model."""
    try:
        # Handle PyTorch 2.6+ compatibility
        import torch.serialization
        try:
            # Try to import the specific class needed
            from ultralytics.nn.tasks import DetectionModel
            torch.serialization.add_safe_globals([DetectionModel])
        except (ImportError, AttributeError):
            # If import fails, we'll use environment variable workaround
            pass
        
        # Set environment variable to handle weights loading
        os.environ['ULTRALYTICS_WEIGHTS_ONLY'] = 'False'
        
        model = YOLO(model_path)
        print(f"Model {model_path} loaded successfully")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

def process_frame(frame, model, conf_threshold=0.25):
    """Process a single frame with YOLOv8 model."""
    results = model(frame, conf=conf_threshold)[0]
    return results

def draw_detections(frame, results):
    """Draw bounding boxes and labels on the frame."""
    annotated_frame = results.plot()
    
    # Add detection summary at the top of the frame
    detected_classes = {}
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        cls_name = results.names[cls_id]
        if cls_name in detected_classes:
            detected_classes[cls_name] += 1
        else:
            detected_classes[cls_name] = 1
    
    summary = "Detected: " + ", ".join(f"{count} {name}" for name, count in detected_classes.items())
    cv2.putText(annotated_frame, summary, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return annotated_frame

def export_detections_to_json(results, output_dir="detections"):
    """Export detection results to a JSON file."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Timestamp for the data
    timestamp = int(time.time() * 1000)
    
    # Extract detection information
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0].item())
        cls_id = int(box.cls[0].item())
        cls_name = results.names[cls_id]
        
        detection = {
            "class": cls_name,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2],
            "timestamp": timestamp
        }
        detections.append(detection)
    
    # Create detection summary
    detection_summary = {
        "timestamp": timestamp,
        "detections": detections,
        "total_objects": len(detections),
        "frame_id": results.path
    }
    
    # Always write to the same file - 'latest_detection.json'
    output_path = os.path.join(output_dir, "latest_detection.json")
    with open(output_path, 'w') as f:
        json.dump(detection_summary, f, indent=2)
    
    return output_path

def main():
    """Main function."""
    args = parse_args()
    
    # Check if a config file exists and load device ID from it
    config_file = "detector_config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                # Override device ID from config if not explicitly provided in command line
                if 'device' in config and args.device == 0:
                    args.device = config['device']
                    print(f"Using camera device {args.device} from configuration file")
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
    
    # Initialize camera
    print(f"Initializing camera device {args.device}...")
    cap = initialize_camera(args.device)
    
    # Get camera properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera resolution: {frame_width}x{frame_height}, FPS: {fps}")
    
    # Initialize YOLOv8 model
    print(f"Loading YOLOv8 model: {args.model}...")
    model = initialize_model(args.model)
    
    # Initialize video writer if saving is enabled
    video_writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.output, fourcc, fps, (frame_width, frame_height)
        )
        print(f"Output video will be saved to {args.output}")
    
    # Display controls
    print("Controls:")
    print("  Press 'q' to quit")
    print("  Press 's' to take a screenshot")
    print("  Press 'j' to force JSON export of current frame")
    
    # Create JSON exports directory if it doesn't exist
    if not os.path.exists(args.json_dir):
        os.makedirs(args.json_dir)
        print(f"Created directory for JSON exports: {args.json_dir}")
    
    # Set target frame rate for JSON exports to 30 frames per second
    target_export_fps = 30
    frame_interval = 1.0 / target_export_fps  # Time between frames in seconds
    last_export_time = 0
    
    print(f"JSON export rate: {target_export_fps} frames per second to {args.json_dir}/")
    
    # Main processing loop
    try:
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.bind("tcp://*:5555")
        
        while True:
            # Read frame from camera
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera")
                break
            
            # Process frame with YOLOv8
            start_time = time.time()
            results = process_frame(frame, model, args.conf)
            process_time = (time.time() - start_time) * 1000
            
            # Get current time
            current_time = time.time()
            
            # Determine if we should export this frame based on the target frame rate
            should_export = (current_time - last_export_time) >= frame_interval
            
            # Export detections to JSON at the target frame rate (30 fps)
            if should_export:
                if len(results.boxes) > 0:
                    json_path = export_detections_to_json(results, args.json_dir)
                    print(f"Exported detections to {json_path}")
                else:
                    # Still write an empty detection file to maintain the frame rate
                    timestamp = int(current_time * 1000)
                    detection_summary = {
                        "timestamp": timestamp,
                        "detections": [],
                        "total_objects": 0,
                        "frame_id": "empty_frame"
                    }
                    output_path = os.path.join(args.json_dir, "latest_detection.json")
                    with open(output_path, 'w') as f:
                        json.dump(detection_summary, f, indent=2)
                
                # Update last export time
                last_export_time = current_time
            
            # Draw detections on frame
            annotated_frame = draw_detections(frame, results)
            
            # Send the annotated frame through ZeroMQ
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            socket.send_string(jpg_as_text)
            
            # Add performance info and export status
            time_until_next = max(0, frame_interval - (time.time() - last_export_time))
            status_text = f"Inference: {process_time:.1f}ms | Next export in: {time_until_next:.3f}s | Export rate: 30 fps"
            cv2.putText(annotated_frame, status_text, (10, frame_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Write frame to output video if saving is enabled
            if args.save and video_writer is not None:
                video_writer.write(annotated_frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_path = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_path, annotated_frame)
                print(f"Screenshot saved to {screenshot_path}")
            elif key == ord('j'):
                # Force JSON export on keypress
                if len(results.boxes) > 0:
                    json_path = export_detections_to_json(results, args.json_dir)
                    last_export_time = current_time
                    print(f"Manually exported detections to {json_path}")
                else:
                    print("No detections to export in current frame")
    
    finally:
        # Clean up
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        print("Application terminated")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")