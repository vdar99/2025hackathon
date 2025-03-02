#!/usr/bin/env python3
"""
Camera Device Preview Tool

This script allows you to view and test multiple camera devices
simultaneously to identify which one is your OVR feed.
"""

import cv2
import argparse
import time
import os
import json
import signal
import sys
from threading import Thread

# Global flag for graceful exit
running = True

class CameraPreview:
    """Class to handle camera preview"""
    
    def __init__(self, device_id):
        self.device_id = device_id
        self.cap = None
        self.thread = None
        self.frame = None
        self.is_running = False
        self.resolution = (0, 0)
        self.fps = 0
        self.window_name = f"Camera {device_id}"
    
    def start(self):
        """Start camera preview"""
        self.cap = cv2.VideoCapture(self.device_id)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.device_id}")
            return False
        
        # Get camera properties
        self.resolution = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera {self.device_id}: {self.resolution[0]}x{self.resolution[1]} @ {self.fps}fps")
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 640, 480)
        
        # Start thread
        self.is_running = True
        self.thread = Thread(target=self._update)
        self.thread.daemon = True
        self.thread.start()
        
        return True
    
    def _update(self):
        """Thread function to continuously get frames"""
        while self.is_running:
            if not self.cap.isOpened():
                print(f"Error: Camera {self.device_id} connection lost")
                break
                
            ret, frame = self.cap.read()
            
            if not ret:
                print(f"Warning: Failed to get frame from camera {self.device_id}")
                time.sleep(0.1)
                continue
            
            # Add device ID and info to frame
            text = f"Camera {self.device_id}: {self.resolution[0]}x{self.resolution[1]}"
            cv2.putText(frame, text, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press corresponding number key to select", (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            self.frame = frame
            
            # Display frame
            cv2.imshow(self.window_name, frame)
            cv2.waitKey(1)
    
    def stop(self):
        """Stop camera preview"""
        self.is_running = False
        
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyWindow(self.window_name)


def signal_handler(sig, frame):
    """Handle Ctrl+C for graceful exit"""
    global running
    print("\nExiting...")
    running = False


def save_selected_device(device_id):
    """Save selected device ID to configuration file"""
    config_file = "detector_config.json"
    config = {}
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load existing config: {e}")
    
    # Update or add device ID
    config['device'] = device_id
    
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Camera device {device_id} saved to configuration")
        return True
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False


def take_screenshot(preview):
    """Take a screenshot from the camera preview"""
    if preview.frame is None:
        print("No frame available for screenshot")
        return
    
    filename = f"camera_{preview.device_id}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, preview.frame)
    print(f"Screenshot saved as {filename}")


def main():
    """Main function"""
    global running
    
    # Set up signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Camera Device Preview Tool')
    parser.add_argument('--devices', type=str, default='0,1,2',
                        help='Comma-separated list of device IDs to preview')
    parser.add_argument('--max', type=int, default=4,
                        help='Maximum number of devices to preview simultaneously')
    args = parser.parse_args()
    
    # Parse device IDs
    try:
        device_ids = [int(x) for x in args.devices.split(',')]
        if not device_ids:
            device_ids = list(range(args.max))
    except ValueError:
        print("Error: Invalid device IDs format")
        return 1
    
    if len(device_ids) > args.max:
        print(f"Warning: Limiting to {args.max} devices for performance")
        device_ids = device_ids[:args.max]
    
    # Start previews
    previews = {}
    for device_id in device_ids:
        preview = CameraPreview(device_id)
        if preview.start():
            previews[device_id] = preview
    
    if not previews:
        print("Error: No cameras available")
        return 1
    
    print("\nPreview started for cameras: " + ", ".join(str(d) for d in previews.keys()))
    print("\nControls:")
    print("  Press 0-9 to select corresponding camera")
    print("  Press 's' + device number to take a screenshot (e.g., 's1' for camera 1)")
    print("  Press 'q' to quit")
    
    # Main loop
    while running and previews:
        key = cv2.waitKey(100) & 0xFF
        
        if key == ord('q'):
            break
        elif ord('0') <= key <= ord('9'):
            # Select camera
            selected = key - ord('0')
            if selected in previews:
                save_selected_device(selected)
                print(f"\nCamera {selected} selected for YOLOv8 detector")
                print(f"Run the detector with: python ovr_yolo_detector.py --device {selected}")
                print("Or simply run: python ovr_yolo_detector.py (it will use the saved configuration)")
            else:
                print(f"Camera {selected} is not available")
        elif key == ord('s'):
            # Prepare for screenshot (need second key)
            print("Press device number to take screenshot")
            screenshot_key = cv2.waitKey(3000) & 0xFF
            if ord('0') <= screenshot_key <= ord('9'):
                device = screenshot_key - ord('0')
                if device in previews:
                    take_screenshot(previews[device])
                else:
                    print(f"Camera {device} is not available")
    
    # Clean up
    for preview in previews.values():
        preview.stop()
    
    cv2.destroyAllWindows()
    print("Preview closed")
    return 0


if __name__ == "__main__":
    sys.exit(main())