#!/usr/bin/env python3
"""
Full-stack application combining OVR YOLOv8 object detection with
speech recognition and TTS for an assistive vision system.
"""
import zmq
import base64
import os
import sys
import json
import time
import threading
import queue
import logging
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import speech_recognition as sr
import pyttsx3
from flask import Flask, render_template, Response, jsonify
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VisionAssistant")

# Global variables
speech_queue = queue.Queue()
response_queue = queue.Queue()
latest_transcript = ""
latest_response = ""
latest_objects = []
latest_frame = None
recognition_active = True
openai_enabled = False

# Initialize OpenAI if API key is available
try:
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    if openai.api_key:
        openai_enabled = True
        logger.info("OpenAI API key found. Description generation enabled.")
    else:
        logger.warning("OpenAI API key not found. Description generation disabled.")
except ImportError:
    logger.warning("OpenAI module not available. Description generation disabled.")

# Initialize Flask app
app = Flask(__name__)

# Initialize Text-to-Speech engine
def init_tts():
    engine = pyttsx3.init()
    # Configure voice properties
    engine.setProperty('rate', 180)  # Speed (words per minute)
    engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
    
    # Try to use a more natural-sounding voice if available
    voices = engine.getProperty('voices')
    for voice in voices:
        # Prefer female voices as they tend to be clearer
        if "female" in voice.name.lower() or "en-us" in voice.id.lower():
            engine.setProperty('voice', voice.id)
            logger.info(f"Using voice: {voice.name}")
            break
    
    return engine

# Speech recognition thread
def speech_recognition_thread():
    global latest_transcript, recognition_active
    """Potentially Problematic"""
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300  # Adjust based on your microphone sensitivity
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8  # Shorter pause for faster response
    
    logger.info("Speech recognition thread started")
    
    while recognition_active:
        try:
            with sr.Microphone() as source:
                logger.info("Listening for commands...")
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=5)
            
            try:
                text = recognizer.recognize_google(audio).lower()
                logger.info(f"Recognized: {text}")
                latest_transcript = text
                speech_queue.put(text)
            except sr.UnknownValueError:
                logger.debug("Could not understand audio")
            except sr.RequestError as e:
                logger.error(f"Speech recognition service error: {e}")
                
        except Exception as e:
            logger.error(f"Error in speech recognition: {e}")
            time.sleep(1)  # Prevent tight loop if there's an error
    
    logger.info("Speech recognition thread stopped")

# Command processing thread
def command_processor_thread(json_dir):
    global latest_response, latest_objects
    
    # tts_engine = init_tts()
    # logger.info("Command processor thread started")
    
    # while recognition_active:
    #     try:
    #         # Get the latest spoken command
    #         command = speech_queue.get(timeout=0.5)
            
    #         # Process different commands
    #         if "what is in front of me" in command or "what do you see" in command:
    #             # Read latest detections from JSON file
    #             detected_objects = get_latest_detections(json_dir)
    #             latest_objects = detected_objects
                
    #             if detected_objects:
    #                 # Count objects by class
    #                 object_counts = {}
    #                 for obj in detected_objects:
    #                     obj_class = obj.get("class", "unknown")
    #                     if obj_class in object_counts:
    #                         object_counts[obj_class] += 1
    #                     else:
    #                         object_counts[obj_class] = 1
                    
    #                 # Format response
    #                 object_list = ", ".join([f"{count} {obj_class}" for obj_class, count in object_counts.items()])
    #                 response = f"I can see {object_list}."
    #             else:
    #                 response = "I don't see any recognizable objects at the moment."
                
    #             # Queue response for TTS
    #             latest_response = response
    #             response_queue.put(response)
                
    #         elif "describe my surroundings" in command or "describe what you see" in command:
    #             # Generate description based on screenshot and detections
    #             response = generate_scene_description(json_dir)
    #             latest_response = response
    #             response_queue.put(response)
                
    #         elif "stop" in command or "exit" in command or "quit" in command:
    #             response = "Shutting down the assistant. Goodbye!"
    #             latest_response = response
    #             response_queue.put(response)
    #             # Don't exit here, just acknowledge the command
                
    #         elif "hello" in command or "hi" in command:
    #             response = "Hello! I'm your vision assistant. How can I help you today?"
    #             latest_response = response
    #             response_queue.put(response)
                
    #         elif "take a screenshot" in command or "capture this" in command:
    #             if latest_frame is not None:
    #                 screenshot_path = save_screenshot(latest_frame)
    #                 response = f"I've captured this view as {screenshot_path}"
    #                 latest_response = response
    #                 response_queue.put(response)
    #             else:
    #                 response = "I couldn't take a screenshot because no camera feed is available."
    #                 latest_response = response
    #                 response_queue.put(response)
                
    #         else:
    #             # Generic response for unrecognized commands
    #             response = f"I heard you say: {command}. You can ask 'what is in front of me' or 'describe my surroundings'."
    #             latest_response = response
    #             response_queue.put(response)
            
    #         # Mark the command as processed
    #         speech_queue.task_done()
            
    #     except queue.Empty:
    #         # No new commands, continue waiting
    #         pass
    #     except Exception as e:
    #         logger.error(f"Error processing command: {e}")
    
    # logger.info("Command processor thread stopped")

# TTS output thread
def tts_output_thread():
    tts_engine = init_tts()
    logger.info("TTS output thread started")
    
    while recognition_active:
        try:
            # Get the latest response to vocalize
            response = response_queue.get(timeout=0.5)
            
            # Speak the response
            logger.info(f"Speaking: {response}")
            tts_engine.say(response)
            tts_engine.runAndWait()
            
            # Mark the response as spoken
            response_queue.task_done()
            
        except queue.Empty:
            # No new responses, continue waiting
            pass
        except Exception as e:
            logger.error(f"Error in TTS output: {e}")
    
    logger.info("TTS output thread stopped")

def video_capture_thread(camera_id):
    global latest_frame
    
    logger.info(f"Starting video capture from camera {camera_id}")
    
    # For backup if ZMQ frames aren't arriving yet
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"Failed to open camera {camera_id}")
    
    while recognition_active:
        frame = None
        
        # If we have a frame from ZMQ, use that
        if latest_frame is not None:
            frame = latest_frame.copy()
        # Otherwise fall back to the camera
        elif cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                time.sleep(0.1)
                continue
        else:
            # If no frame is available, create a blank one
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
            cv2.putText(frame, "Waiting for video feed...", (50, 240), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Add timestamp to frame
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add speech recognition status
        cv2.putText(frame, f"Heard: {latest_transcript}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add response text
        y_pos = 90
        wrapped_text = [latest_response[i:i+50] for i in range(0, len(latest_response), 50)]
        for line in wrapped_text:
            cv2.putText(frame, line, (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 30
        
        # Encode the frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        # Convert to bytes for Flask
        frame_bytes = buffer.tobytes()
        
        # Yield the frame for the Flask video feed
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    # Clean up
    if cap.isOpened():
        cap.release()
    logger.info("Video capture thread stopped")

# Helper function to get latest detections from JSON file
"""Potentially Problematic"""
def get_latest_detections(json_dir="detections"):
    try:
        # Find the most recent JSON file in the directory
        json_files = list(Path(json_dir).glob("*.json"))
        if not json_files:
            logger.warning(f"No JSON files found in {json_dir}")
            return []
        
        # Sort by modification time (most recent first)
        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Reading detections from {latest_file}")
        
        # Read the JSON file
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        # Extract detections
        detections = data.get("detections", [])
        return detections
        
    except Exception as e:
        logger.error(f"Error reading detection data: {e}")
        return []

# Helper function to save a screenshot
def save_screenshot(frame, output_dir="screenshots"):
    try:
        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = int(time.time())
        filename = os.path.join(output_dir, f"screenshot_{timestamp}.jpg")
        
        # Save the screenshot
        cv2.imwrite(filename, frame)
        logger.info(f"Saved screenshot to {filename}")
        zmq_frame_receiver()
        return filename
    except Exception as e:
        logger.error(f"Error saving screenshot: {e}")
        return None

# Modify the zmq_frame_receiver function
def zmq_frame_receiver():
    """Receives frames from ovr_yolo_detector.py over ZeroMQ."""
    global latest_frame, recognition_active

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5555")
    socket.setsockopt_string(zmq.SUBSCRIBE, '')
    
    # Set socket to non-blocking mode
    socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout

    logger.info("ZMQ receiver started")
    
    while recognition_active:
        try:
            jpg_as_text = socket.recv_string()  # Will timeout after 100ms if no message
            jpg_as_bytes = base64.b64decode(jpg_as_text)
            jpg_np = np.frombuffer(jpg_as_bytes, dtype=np.uint8)
            frame = cv2.imdecode(jpg_np, cv2.IMREAD_COLOR)

            if frame is not None:
                latest_frame = frame  # Update global frame
                logger.info("Received frame from detector")
        except zmq.Again:
            # Timeout occurred, just continue
            pass
        except Exception as e:
            logger.error(f"Error in ZMQ receiver: {e}")
            time.sleep(0.1)  # Avoid tight loop in case of errors
    
    logger.info("ZMQ receiver stopped")
    socket.close()
    context.term()

# Helper function to generate a scene description using OpenAI or basic logic
def generate_scene_description(json_dir="detections"):
    global latest_frame, latest_objects
    
    # Take a screenshot if we have a frame
    screenshot_path = None
    if latest_frame is not None:
        screenshot_path = save_screenshot(latest_frame)
    
    # Get the latest detections
    detections = latest_objects
    if not detections:
        detections = get_latest_detections(json_dir)
        latest_objects = detections
    
    # If no detections, return simple message
    if not detections:
        return "I don't see any recognizable objects in view at the moment."
    
    # Check if OpenAI is available
    if openai_enabled:
        try:
            # Format the detection information
            detection_text = ""
            object_counts = {}
            
            for obj in detections:
                obj_class = obj.get("class", "unknown")
                confidence = obj.get("confidence", 0)
                
                # Count objects by class
                if obj_class in object_counts:
                    object_counts[obj_class] += 1
                else:
                    object_counts[obj_class] = 1
                
                # Add position information
                bbox = obj.get("bbox", [0, 0, 0, 0])
                if len(bbox) == 4:
                    # Calculate center point
                    x_center = (bbox[0] + bbox[2]) / 2
                    y_center = (bbox[1] + bbox[3]) / 2
                    
                    # Determine position (simple quadrant-based)
                    horiz = "right side" if x_center > 0.5 else "left side"
                    vert = "bottom" if y_center > 0.5 else "top"
                    
                    detection_text += f"- {obj_class} (confidence: {confidence:.2f}) on the {vert} {horiz}\n"
            
            # Add summary counts
            detection_text += "Summary: "
            detection_text += ", ".join([f"{count} {obj_class}" for obj_class, count in object_counts.items()])
            
            # Create the prompt for OpenAI
            prompt = f"""
            You are a vision assistant helping a user understand their surroundings.
            Based on the object detection data below, create a natural, conversational description
            of what's likely in the scene. Be concise but informative, mentioning spatial relationships
            between objects when possible. The description should sound like someone helpfully
            describing what they see to another person.
            
            Detection data:
            {detection_text}
            """
            
            # Call OpenAI API
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that describes scenes based on object detection data."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150
            )
            
            # Extract the description from the response
            description = response.choices[0].message.content.strip()
            logger.info(f"Generated description: {description}")
            
            return description
            
        except Exception as e:
            logger.error(f"Error generating scene description with OpenAI: {e}")
            # Fall back to basic description
    
    # Basic description approach (when OpenAI is not available or fails)
    try:
        # Group objects by position
        left_objects = {}
        center_objects = {}
        right_objects = {}
        
        for obj in detections:
            obj_class = obj.get("class", "unknown")
            bbox = obj.get("bbox", [0, 0, 0, 0])
            
            # Determine position
            if len(bbox) == 4:
                x_center = (bbox[0] + bbox[2]) / 2
                width = max(bbox[2] - bbox[0], 1)  # Prevent division by zero
                relative_x = x_center / width
                
                # Assign to left, center, or right
                if relative_x < 0.33:
                    if obj_class in left_objects:
                        left_objects[obj_class] += 1
                    else:
                        left_objects[obj_class] = 1
                elif relative_x > 0.66:
                    if obj_class in right_objects:
                        right_objects[obj_class] += 1
                    else:
                        right_objects[obj_class] = 1
                else:
                    if obj_class in center_objects:
                        center_objects[obj_class] += 1
                    else:
                        center_objects[obj_class] = 1
            else:
                # If no bounding box, just count the object
                if obj_class in center_objects:
                    center_objects[obj_class] += 1
                else:
                    center_objects[obj_class] = 1
        
        # Build the description
        description_parts = []
        
        if left_objects:
            left_str = ", ".join([f"{count} {obj_class}" for obj_class, count in left_objects.items()])
            description_parts.append(f"On the left, I can see {left_str}")
        
        if center_objects:
            center_str = ", ".join([f"{count} {obj_class}" for obj_class, count in center_objects.items()])
            description_parts.append(f"In the center, I can see {center_str}")
        
        if right_objects:
            right_str = ", ".join([f"{count} {obj_class}" for obj_class, count in right_objects.items()])
            description_parts.append(f"On the right, I can see {right_str}")
        
        # Combine all parts
        if description_parts:
            return ". ".join(description_parts) + "."
        else:
            # Fallback if positional grouping failed
            all_objects = {}
            for obj in detections:
                obj_class = obj.get("class", "unknown")
                if obj_class in all_objects:
                    all_objects[obj_class] += 1
                else:
                    all_objects[obj_class] = 1
            
            objects_str = ", ".join([f"{count} {obj_class}" for obj_class, count in all_objects.items()])
            return f"I can see {objects_str}."
            
    except Exception as e:
        logger.error(f"Error generating basic scene description: {e}")
        # Final fallback
        detected_classes = list(set([obj.get("class", "unknown") for obj in detections]))
        return f"I can see {', '.join(detected_classes)}."

# Flask routes for the web interface
@app.route('/')
def index():
    """Home page route"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(
        video_capture_thread(app.config['CAMERA_ID']),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/status')
def api_status():
    """API endpoint for getting the current status"""
    return jsonify({
        'transcript': latest_transcript,
        'response': latest_response,
        'objects': latest_objects
    })

@app.route('/api/screenshot', methods=['POST'])
def take_api_screenshot():
    """API endpoint for taking a screenshot"""
    if latest_frame is not None:
        screenshot_path = save_screenshot(latest_frame)
        return jsonify({
            'status': 'success',
            'path': screenshot_path,
            'message': f'Screenshot saved to {screenshot_path}'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'No frame available for screenshot'
        }), 400
    
def main():
    """Main function"""
    global recognition_active
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Vision Assistant with Speech Recognition')
    parser.add_argument('--device', type=int, default=0, 
                        help='Camera device ID (default: 0)')
    parser.add_argument('--json-dir', type=str, default='detections',
                        help='Directory containing detection JSON files (default: "detections")')
    parser.add_argument('--port', type=int, default=3000,
                        help='Port for the web interface (default: 3000)')
    args = parser.parse_args()
    
    try:
        # Configure Flask app
        app.config['CAMERA_ID'] = 1
        app.config['JSON_DIR'] = args.json_dir
        
        # Create necessary directories
        os.makedirs("screenshots", exist_ok=True)
        os.makedirs(args.json_dir, exist_ok=True)
        
        # Start ZMQ thread to receive frames from detector
        zmq_thread = threading.Thread(target=zmq_frame_receiver)
        zmq_thread.daemon = True
        zmq_thread.start()
        
        # Start the command processor thread
        processor_thread = threading.Thread(target=command_processor_thread, args=(args.json_dir,))
        processor_thread.daemon = True
        processor_thread.start()
        
        # Welcome message
        welcome_msg = "Vision assistant is now running."
        print(welcome_msg)
        response_queue.put(welcome_msg)
        
        # Start the Flask web server
        logger.info(f"Starting web interface on port {args.port}")
        print(f"\nOpen your browser and go to: http://localhost:{args.port} to see the interface")
        app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
    finally:
        # Clean up and exit
        recognition_active = False
        logger.info("Vision assistant shutdown complete")


if __name__ == "__main__":
    main()