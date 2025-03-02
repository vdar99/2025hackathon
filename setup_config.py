#!/usr/bin/env python3
"""
Setup and configuration utility for OVR YOLOv8 detector
"""

import os
import sys
import subprocess
import json

CONFIG_FILE = "detector_config.json"
DEFAULT_CONFIG = {
    "model": "yolov8n.pt",
    "confidence": 0.25,
    "device": 0,
    "save_output": False,
    "output_path": "output.mp4",
    "classes_of_interest": []
}

def check_environment():
    """Check if the required packages are installed."""
    required_packages = [
        "ultralytics",
        "opencv-python",
        "numpy",
        "pillow",
        "PyObjC"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.split('-')[0])  # Handle opencv-python
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        
        install = input("Install missing packages? (y/n): ")
        if install.lower() == 'y':
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("Packages installed successfully.")
        else:
            print("Please install the missing packages manually.")
            sys.exit(1)
    else:
        print("All required packages are installed.")

def download_model(model_name="yolov8n.pt"):
    """Download YOLOv8 model if not exists."""
    if os.path.exists(model_name):
        print(f"Model {model_name} already exists.")
        return
    
    print(f"Downloading model {model_name}...")
    try:
        # Add safe globals for PyTorch 2.6+ compatibility
        import torch.serialization
        try:
            # Try to import the specific class needed
            from ultralytics.nn.tasks import DetectionModel
            torch.serialization.add_safe_globals([DetectionModel])
        except (ImportError, AttributeError):
            # If import fails, we'll try a more general approach
            print("Note: Using a workaround for PyTorch 2.6+ compatibility")
            pass
            
        # Set environment variable to handle weights loading
        os.environ['ULTRALYTICS_WEIGHTS_ONLY'] = 'False'
        
        from ultralytics import YOLO
        model = YOLO(model_name)
        print(f"Model {model_name} downloaded successfully.")
    except Exception as e:
        print(f"Failed to download model: {e}")
        print("\nTroubleshooting steps:")
        print("1. Manually download the model using:")
        print(f"   from ultralytics import YOLO; YOLO('{model_name}')")
        print("2. If that fails, try installing an older version of PyTorch:")
        print("   pip install torch==2.0.1 torchvision==0.15.2")
        print("3. Another option is to create a new Python environment:")
        print("   python -m venv yolo_env")
        print("   source yolo_env/bin/activate  # On macOS/Linux")
        print("   pip install -r requirements.txt")
        sys.exit(1)

def list_available_cameras():
    """List available camera devices."""
    import cv2
    max_to_check = 5  # Check first 5 cameras
    
    print("Checking available camera devices:")
    available_cameras = []
    
    for i in range(max_to_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                print(f"  Camera {i}: Available ({resolution[0]}x{resolution[1]})")
                available_cameras.append(i)
            else:
                print(f"  Camera {i}: Connected but unable to read frames")
            cap.release()
        else:
            print(f"  Camera {i}: Not available")
    
    return available_cameras

def create_config():
    """Create or update configuration file."""
    config = DEFAULT_CONFIG.copy()
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                existing_config = json.load(f)
                config.update(existing_config)
            print(f"Loaded existing configuration from {CONFIG_FILE}")
        except Exception as e:
            print(f"Error reading existing config: {e}")
    
    print("\nAvailable YOLOv8 models:")
    models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
    for i, model in enumerate(models):
        print(f"  {i+1}. {model}")
    
    model_idx = input(f"Select model (1-{len(models)}, default: {models.index('yolov8n.pt')+1}): ")
    if model_idx.strip() and model_idx.isdigit() and 1 <= int(model_idx) <= len(models):
        config["model"] = models[int(model_idx) - 1]
    
    confidence = input(f"Confidence threshold (0.0-1.0, default: {config['confidence']}): ")
    if confidence.strip() and confidence.replace('.', '', 1).isdigit():
        conf_val = float(confidence)
        if 0 <= conf_val <= 1:
            config["confidence"] = conf_val
    
    print("\nChecking available cameras...")
    available_cameras = list_available_cameras()
    
    if available_cameras:
        device = input(f"Select camera device ({', '.join(map(str, available_cameras))}, default: {config['device']}): ")
        if device.strip() and device.isdigit() and int(device) in available_cameras:
            config["device"] = int(device)
    else:
        print("No cameras detected. Using default device ID.")
    
    save_output = input(f"Save output video? (y/n, default: {'y' if config['save_output'] else 'n'}): ")
    if save_output.strip().lower() in ['y', 'n']:
        config["save_output"] = (save_output.lower() == 'y')
    
    if config["save_output"]:
        output_path = input(f"Output video path (default: {config['output_path']}): ")
        if output_path.strip():
            config["output_path"] = output_path
    
    # Save configuration
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Configuration saved to {CONFIG_FILE}")
    print(f"\nIMPORTANT: Your selected camera ID ({config['device']}) has been saved to the configuration.")
    return config

def main():
    """Main function."""
    print("=" * 50)
    print("OVR YOLOv8 Detector Setup Utility")
    print("=" * 50)
    
    # Check environment
    print("\nChecking environment...")
    check_environment()
    
    # Create configuration
    print("\nConfiguring detector...")
    config = create_config()
    
    # Download model if needed
    print("\nChecking model...")
    download_model(config["model"])
    
    print("\nSetup completed successfully!")
    print(f"Run the detector with: python ovr_yolo_detector.py --model {config['model']} --conf {config['confidence']} --device {config['device']} {'--save' if config['save_output'] else ''}")

if __name__ == "__main__":
    main()