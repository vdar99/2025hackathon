#!/bin/bash
# Force a specific camera device ID for the YOLOv8 detector

# Default camera ID if not provided
CAMERA_ID=${1:-1}

echo "Setting up YOLOv8 detector to use camera device ID: $CAMERA_ID"

# Check if the configuration file exists
CONFIG_FILE="detector_config.json"
if [ -f "$CONFIG_FILE" ]; then
    # Backup the existing config
    cp "$CONFIG_FILE" "${CONFIG_FILE}.bak"
    echo "Backed up existing configuration to ${CONFIG_FILE}.bak"
    
    # Update the device ID in the config file
    # This uses temporary files since macOS might not have the right version of sed
    cat "$CONFIG_FILE" | sed "s/\"device\": [0-9]*/\"device\": $CAMERA_ID/" > "${CONFIG_FILE}.tmp"
    mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
    
    echo "Updated configuration file with camera device ID: $CAMERA_ID"
else
    # Create a new minimal configuration
    echo "{
  \"model\": \"yolov8n.pt\",
  \"confidence\": 0.25,
  \"device\": $CAMERA_ID,
  \"save_output\": false,
  \"output_path\": \"output.mp4\"
}" > "$CONFIG_FILE"
    
    echo "Created new configuration file with camera device ID: $CAMERA_ID"
fi

# Run the detector with the explicit device ID to ensure it's used
echo "Running YOLOv8 detector with camera device $CAMERA_ID..."
echo "Command: python ovr_yolo_detector.py --device $CAMERA_ID"
python ovr_yolo_detector.py --device $CAMERA_ID