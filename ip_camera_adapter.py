#!/usr/bin/env python3
"""
IP Camera Adapter for OVR YOLOv8 Object Detector

This script allows connecting to an IP camera or RTSP stream
from an OVR device and using it with the YOLOv8 detector.
"""

import cv2
import argparse
import time
import os
import sys
import signal
import json
from threading import Thread
import queue

# Global flag for graceful exit
running = True

class IPCameraStream:
    """Class to handle IP camera streaming with buffering"""
    
    def __init__(self, url, buffer_size=10):
        self.url = url
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.thread = None
        self.cap = None
        self.is_running = False
        self.fps = 0
        self.resolution = (0, 0)
        self.frame_count = 0
        self.start_time = time.time()
    
    def start(self):
        """Start the camera stream in a separate thread"""
        self.is_running = True
        self.thread = Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self
    
    def _update(self):
        """Update thread function to continuously get frames"""
        self.cap = cv2.VideoCapture(self.url)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open stream at {self.url}")
            self.is_running = False
            return
        
        # Get camera properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.resolution = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        
        print(f"Connected to stream: {self.resolution[0]}x{self.resolution[1]} @ {self.fps}fps")
        
        # Read frames continuously
        while self.is_running:
            if not self.cap.isOpened():
                print("Error: Stream connection lost. Attempting to reconnect...")
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.url)
                continue
                
            ret, frame = self.cap.read()
            
            if not ret:
                print("Warning: Failed to get frame, retrying...")
                time.sleep(0.1)
                continue
                
            # Calculate actual FPS
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed >= 5.0:  # Update FPS every 5 seconds
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.start_time = time.time()
            
            # If buffer full, remove oldest frame
            if self.buffer.full():
                try:
                    self.buffer.get_nowait()
                except queue.Empty:
                    pass
            
            # Add new frame to buffer
            try:
                self.buffer.put(frame, block=False)
            except queue.Full:
                pass
    
    def read(self):
        """Read a frame from the buffer"""
        if self.buffer.empty():
            return False, None
        
        frame = self.buffer.get()
        return True, frame
    
    def get_info(self):
        """Get information about the stream"""
        return {
            "url": self.url,
            "resolution": self.resolution,
            "fps": self.fps,
            "buffer_size": self.buffer.maxsize,
            "buffer_used": self.buffer.qsize(),
            "running": self.is_running
        }
    
    def release(self):
        """Stop the stream and release resources"""
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


def save_camera_config(url, config_file="ip_camera_config.json"):
    """Save IP camera configuration to file"""
    config = {"url": url, "last_used": time.strftime("%Y-%m-%d %H:%M:%S")}
    
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Camera configuration saved to {config_file}")
    except Exception as e:
        print(f"Error saving camera configuration: {e}")


def load_camera_config(config_file="ip_camera_config.json"):
    """Load IP camera configuration from file"""
    if not os.path.exists(config_file):
        return None
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config.get("url")
    except Exception as e:
        print(f"Error loading camera configuration: {e}")
        return None


def test_stream(url, timeout=10):
    """Test if IP camera stream is accessible"""
    print(f"Testing stream: {url}")
    cap = cv2.VideoCapture(url)
    
    if not cap.isOpened():
        print("Error: Could not open stream")
        return False
    
    # Try to get a frame with timeout
    start_time = time.time()
    frame_received = False
    
    while time.time() - start_time < timeout:
        ret, frame = cap.read()
        if ret:
            frame_received = True
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Stream accessible: {width}x{height} @ {fps}fps")
            break
        time.sleep(0.5)
        print("Waiting for frame...")
    
    cap.release()
    
    if not frame_received:
        print("Error: No frames received within timeout period")
        return False
    
    return True


def create_virtual_device(port=8080):
    """
    Create a virtual video device using ffmpeg that serves the IP camera stream.
    Returns instructions for client usage.
    """
    # This is a placeholder function - implementing virtual video devices on macOS
    # requires additional software like CamTwist or OBS Studio
    print("\nVirtual video device creation on macOS requires additional software:")
    print("1. Install OBS Studio (https://obsproject.com)")
    print("2. Add your IP Camera as a 'Media Source' in OBS")
    print("3. Install OBS Virtual Camera plugin")
    print("4. Start Virtual Camera from OBS")
    print("5. Then select the OBS Virtual Camera in the YOLOv8 detector")
    
    return {
        "instructions": "Use OBS Studio with Virtual Camera plugin",
        "camera_id": "Use device ID from list_video_sources.py after setup"
    }


def signal_handler(sig, frame):
    """Handle Ctrl+C for graceful exit"""
    global running
    print("\nExiting...")
    running = False


def main():
    """Main function"""
    global running
    
    # Set up signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='IP Camera Adapter for OVR YOLOv8 Detector')
    parser.add_argument('--url', type=str, default='',
                        help='URL of the IP camera stream')
    parser.add_argument('--rtsp', type=str, default='',
                        help='RTSP URL of the camera stream')
    parser.add_argument('--test', action='store_true',
                        help='Test the camera stream and exit')
    parser.add_argument('--buffer', type=int, default=10,
                        help='Frame buffer size (default: 10)')
    parser.add_argument('--virtual', action='store_true',
                        help='Create a virtual video device from the IP camera')
    args = parser.parse_args()
    
    # Determine the URL to use
    url = args.url
    if args.rtsp:
        url = args.rtsp
    
    # If no URL provided, try to load from config
    if not url:
        config_url = load_camera_config()
        if config_url:
            print(f"Loaded camera URL from config: {config_url}")
            url = config_url
        else:
            # Prompt user for URL if not provided
            print("Please enter the IP camera URL or RTSP stream address.")
            print("Examples:")
            print("  - RTSP: rtsp://username:password@192.168.1.100:554/stream")
            print("  - HTTP: http://192.168.1.100:8080/video")
            url = input("URL: ").strip()
    
    # Validate URL
    if not url:
        print("Error: No camera URL provided")
        return 1
    
    # Save the configuration
    save_camera_config(url)
    
    # Test the stream if requested
    if args.test:
        if test_stream(url):
            print("Stream test successful")
            return 0
        else:
            print("Stream test failed")
            return 1
    
    # Create virtual device if requested
    if args.virtual:
        create_virtual_device()
        return 0
    
    # Start the stream
    print(f"Starting IP camera stream: {url}")
    stream = IPCameraStream(url, buffer_size=args.buffer).start()
    
    # Display window
    cv2.namedWindow("OVR IP Camera Feed", cv2.WINDOW_NORMAL)
    
    # Create a help message
    help_msg = [
        "IP Camera Adapter - Controls:",
        "Q: Quit",
        "S: Save frame",
        "I: Show stream info",
        "H: Toggle this help"
    ]
    show_help = True
    
    # Main loop
    frame_count = 0
    start_time = time.time()
    actual_fps = 0
    
    # Instructions for YOLOv8 detector
    print("\nTo use this IP camera with the YOLOv8 detector:")
    print("1. Install OBS Studio and set up the Virtual Camera as suggested")
    print("2. Run list_video_sources.py to find the virtual camera ID")
    print("3. Run the detector with:")
    print("   python ovr_yolo_detector.py --device [virtual_camera_id]")
    
    while running:
        ret, frame = stream.read()
        
        if not ret:
            print("Waiting for frames...")
            time.sleep(0.1)
            continue
        
        # Calculate actual FPS
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 3.0:  # Update FPS every 3 seconds
            actual_fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()
        
        # Draw FPS counter
        cv2.putText(frame, f"FPS: {actual_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw help if enabled
        if show_help:
            y_pos = 70
            for line in help_msg:
                cv2.putText(frame, line, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_pos += 30
        
        # Show frame
        cv2.imshow("OVR IP Camera Feed", frame)
        
        # Process key commands
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            filename = f"ovr_capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Frame saved as {filename}")
        elif key == ord('i'):
            # Show stream info
            info = stream.get_info()
            print("\nStream Information:")
            for k, v in info.items():
                print(f"  {k}: {v}")
        elif key == ord('h'):
            # Toggle help
            show_help = not show_help
    
    # Clean up
    stream.release()
    cv2.destroyAllWindows()
    print("IP camera adapter closed")
    return 0


if __name__ == "__main__":
    sys.exit(main())