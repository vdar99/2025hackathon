#!/usr/bin/env python3
"""
Utility to fix PyTorch 2.6+ model loading issues with YOLOv8
"""

import os
import torch
import sys

def fix_pytorch_loading():
    """
    Add necessary safe globals to PyTorch serialization to allow loading YOLOv8 models.
    This addresses the weights_only=True default in PyTorch 2.6+.
    """
    try:
        from ultralytics.nn.tasks import DetectionModel, SegmentationModel, PoseModel, ClassificationModel
        
        # Add all possible model classes to safe globals
        safe_classes = [
            DetectionModel, 
            SegmentationModel, 
            PoseModel, 
            ClassificationModel
        ]
        
        # Register safe globals
        torch.serialization.add_safe_globals(safe_classes)
        
        # Alternative: Set environment variable
        os.environ['ULTRALYTICS_WEIGHTS_ONLY'] = 'False'
        
        print("PyTorch model loading fix applied successfully.")
        return True
    except Exception as e:
        print(f"Warning: Could not apply PyTorch fix: {e}")
        print("Will try alternative methods during model loading.")
        return False

def download_yolo_model(model_name="yolov8n.pt"):
    """
    Force download a YOLO model with the proper fixes for PyTorch 2.6+.
    """
    try:
        # Apply fix first
        fix_pytorch_loading()
        
        # Force download model
        from ultralytics import YOLO
        print(f"Downloading model {model_name}...")
        model = YOLO(model_name)
        print(f"Model {model_name} downloaded successfully to {os.path.abspath(model_name)}")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

if __name__ == "__main__":
    # Check if a model name was provided
    model_name = "yolov8n.pt"  # Default
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    
    # Apply fix and download model
    fix_pytorch_loading()
    success = download_yolo_model(model_name)
    
    if not success:
        print("\nAlternative solutions to try:")
        print("1. Downgrade PyTorch:")
        print("   pip install torch==2.0.1 torchvision==0.15.2")
        print("\n2. Use a clean environment:")
        print("   python -m venv yolo_env")
        print("   source yolo_env/bin/activate  # On macOS/Linux")
        print("   pip install ultralytics torch==2.0.1 torchvision==0.15.2")
        print("\n3. Use the official Ultralytics Docker image:")
        print("   docker pull ultralytics/ultralytics:latest")
        sys.exit(1)
