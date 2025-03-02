#!/usr/bin/env python3
"""
Utility to list all available video sources on macOS,
including detailed information about each source.
"""

import cv2
import subprocess
import sys
import time
import json
import os
from pprint import pprint

def run_avfoundation_device_list():
    """
    Use ffmpeg to list AVFoundation devices on macOS.
    This might show more devices than OpenCV alone.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        # ffmpeg outputs to stderr, not stdout
        output = result.stderr
        
        print("\n===== FFmpeg AVFoundation Devices =====")
        # Extract and print the device listing section
        if "AVFoundation video devices" in output:
            lines = output.split('\n')
            capture = False
            for line in lines:
                if "AVFoundation video devices" in line:
                    capture = True
                    print(line)
                elif "AVFoundation audio devices" in line:
                    capture = False
                elif capture:
                    print(line)
        else:
            print("No AVFoundation devices found in ffmpeg output")
            
    except FileNotFoundError:
        print("FFmpeg not found. Install with: brew install ffmpeg")
    except Exception as e:
        print(f"Error running ffmpeg: {e}")

def test_camera_stream(device_id, timeout=3):
    """
    Test if a camera stream can be opened and frame retrieved.
    Returns resolution, FPS and status.
    """
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        return None, None, "Failed to open"
    
    # Try to get a frame with timeout
    start_time = time.time()
    frame_received = False
    
    while time.time() - start_time < timeout:
        ret, frame = cap.read()
        if ret:
            frame_received = True
            break
        time.sleep(0.1)
    
    # Get camera properties
    if frame_received:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return (width, height), fps, "Working"
    else:
        cap.release()
        return None, None, "No frames received"

def get_camera_properties(device_id):
    """Get all available properties of a camera device."""
    properties = {}
    cap = cv2.VideoCapture(device_id)
    
    if not cap.isOpened():
        return {"status": "Failed to open"}
    
    # Standard properties to check
    prop_ids = [
        (cv2.CAP_PROP_FRAME_WIDTH, "Width"),
        (cv2.CAP_PROP_FRAME_HEIGHT, "Height"),
        (cv2.CAP_PROP_FPS, "FPS"),
        (cv2.CAP_PROP_FOURCC, "Codec"),
        (cv2.CAP_PROP_BRIGHTNESS, "Brightness"),
        (cv2.CAP_PROP_CONTRAST, "Contrast"),
        (cv2.CAP_PROP_SATURATION, "Saturation"),
        (cv2.CAP_PROP_HUE, "Hue"),
        (cv2.CAP_PROP_GAIN, "Gain"),
        (cv2.CAP_PROP_CONVERT_RGB, "ConvertRGB"),
        (cv2.CAP_PROP_BACKLIGHT, "Backlight"),
        (cv2.CAP_PROP_PAN, "Pan"),
        (cv2.CAP_PROP_TILT, "Tilt"),
        (cv2.CAP_PROP_ROLL, "Roll"),
        (cv2.CAP_PROP_IRIS, "Iris"),
        (cv2.CAP_PROP_FOCUS, "Focus"),
        (cv2.CAP_PROP_ZOOM, "Zoom")
    ]
    
    for prop_id, prop_name in prop_ids:
        value = cap.get(prop_id)
        properties[prop_name] = value
    
    # Try to read a frame
    ret, frame = cap.read()
    properties["Frame received"] = "Yes" if ret else "No"
    
    if ret:
        # Save a sample frame for this camera
        filename = f"camera_{device_id}_sample.jpg"
        cv2.imwrite(filename, frame)
        properties["Sample frame"] = filename
    
    cap.release()
    return properties

def list_cameras():
    """List all available camera devices with detailed information."""
    # Check a reasonable number of camera indices
    max_cameras_to_check = 10
    camera_info = {}
    
    print("\n===== Checking OpenCV Camera Sources =====")
    print(f"Testing camera IDs 0 to {max_cameras_to_check-1}...")
    
    for i in range(max_cameras_to_check):
        # Basic test first
        resolution, fps, status = test_camera_stream(i)
        
        if status == "Working":
            print(f"Camera {i}: Available - Resolution: {resolution}, FPS: {fps}")
            # Get detailed properties
            props = get_camera_properties(i)
            camera_info[i] = props
        else:
            print(f"Camera {i}: {status}")
    
    # Save detailed info to JSON file
    with open("camera_details.json", "w") as f:
        json.dump(camera_info, f, indent=2)
    
    print(f"\nDetailed camera information saved to 'camera_details.json'")
    
    # For any working cameras, we provide the sample frame location
    print("\nSample frames saved for working cameras:")
    for cam_id, props in camera_info.items():
        if "Sample frame" in props:
            print(f"Camera {cam_id}: {props['Sample frame']}")
            
    return camera_info

def main():
    """Main function."""
    print("==== Video Source Scanner ====")
    print("This utility will check for available video sources on your Mac")
    print("and provide detailed information about each source.")
    
    # List cameras using OpenCV
    camera_info = list_cameras()
    
    # Use ffmpeg to get AVFoundation devices (macOS specific)
    run_avfoundation_device_list()
    
    # Summary
    print("\n===== Summary =====")
    if camera_info:
        print(f"Found {len(camera_info)} accessible camera sources.")
        print("To use a specific camera with the OVR YOLOv8 detector, run:")
        for cam_id in camera_info:
            print(f"  python ovr_yolo_detector.py --device {cam_id}")
    else:
        print("No accessible camera sources found.")
        print("Check your OVR device connections and drivers.")
    
    print("\nIf your OVR source is not listed, you may need to:")
    print("1. Install proper virtual camera drivers for your OVR device")
    print("2. Use a capture device with the appropriate drivers")
    print("3. Set up an IP camera stream from your OVR system")

if __name__ == "__main__":
    main()