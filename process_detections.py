#!/usr/bin/env python3
"""
Detection JSON Processor for OVR YOLOv8 Object Detector

This utility processes JSON detection files generated by the OVR YOLOv8 detector.
It can be used to:
1. Monitor a directory for new detection files
2. Process detection data in real-time or in batch
3. Generate statistics and visualizations of detected objects
"""

import os
import json
import time
import argparse
import glob
from datetime import datetime
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class DetectionProcessor:
    """Process YOLOv8 detection JSON files"""
    
    def __init__(self, input_dir="detections", archive_dir=None):
        self.input_dir = input_dir
        self.archive_dir = archive_dir
        
        # Create archive directory if specified
        if archive_dir and not os.path.exists(archive_dir):
            os.makedirs(archive_dir)
        
        # Statistics storage
        self.detection_counts = defaultdict(int)  # Count by class
        self.confidence_values = defaultdict(list)  # Confidence by class
        self.detections_over_time = []  # Timestamp and counts
        self.total_processed = 0
    
    def process_file(self, file_path):
        """Process a single detection JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract timestamp
            timestamp = data.get('timestamp', int(time.time() * 1000))
            dt = datetime.fromtimestamp(timestamp / 1000)
            
            # Count detections
            class_counts = Counter()
            for detection in data.get('detections', []):
                class_name = detection.get('class', 'unknown')
                confidence = detection.get('confidence', 0)
                
                # Update statistics
                self.detection_counts[class_name] += 1
                self.confidence_values[class_name].append(confidence)
                class_counts[class_name] += 1
            
            # Record time-based data
            self.detections_over_time.append({
                'timestamp': timestamp,
                'datetime': dt,
                'counts': dict(class_counts),
                'total': sum(class_counts.values())
            })
            
            self.total_processed += 1
            
            # Print summary
            print(f"Processed: {os.path.basename(file_path)} | "
                  f"Objects: {sum(class_counts.values())} | "
                  f"Classes: {', '.join(f'{k}:{v}' for k, v in class_counts.items())}")
            
            # Move to archive if needed
            if self.archive_dir:
                archive_path = os.path.join(self.archive_dir, os.path.basename(file_path))
                os.rename(file_path, archive_path)
            
            return True
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False
    
    def process_directory(self, delete_processed=False):
        """Process all JSON files in the input directory"""
        file_pattern = os.path.join(self.input_dir, "*.json")
        json_files = glob.glob(file_pattern)
        
        if not json_files:
            print(f"No JSON files found in {self.input_dir}")
            return 0
        
        print(f"Processing {len(json_files)} JSON files...")
        processed_count = 0
        
        for file_path in sorted(json_files):
            if self.process_file(file_path):
                processed_count += 1
                
                # Delete if requested and not archiving
                if delete_processed and not self.archive_dir:
                    os.remove(file_path)
        
        print(f"Processed {processed_count} files")
        return processed_count
    
    def generate_statistics(self):
        """Generate statistics from processed detection data"""
        stats = {
            'total_files_processed': self.total_processed,
            'total_objects_detected': sum(self.detection_counts.values()),
            'detection_by_class': dict(self.detection_counts),
            'average_confidence': {
                cls: sum(values) / len(values) if values else 0 
                for cls, values in self.confidence_values.items()
            }
        }
        
        # Most recent detections (last 10)
        recent = sorted(self.detections_over_time, key=lambda x: x['timestamp'], reverse=True)[:10]
        stats['recent_detections'] = recent
        
        return stats
    
    def plot_detection_trends(self, output_file="detection_trends.png"):
        """Generate a plot of detections over time"""
        if not self.detections_over_time:
            print("No detection data available for plotting")
            return
        
        # Sort by timestamp
        sorted_data = sorted(self.detections_over_time, key=lambda x: x['timestamp'])
        
        # Extract datetime and counts
        datetimes = [entry['datetime'] for entry in sorted_data]
        
        # Get all unique classes
        all_classes = set()
        for entry in sorted_data:
            all_classes.update(entry['counts'].keys())
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot total counts
        totals = [entry['total'] for entry in sorted_data]
        plt.plot(datetimes, totals, 'k-', label='Total', linewidth=2)
        
        # Plot individual classes
        for cls in all_classes:
            counts = [entry['counts'].get(cls, 0) for entry in sorted_data]
            plt.plot(datetimes, counts, '--', label=cls)
        
        plt.xlabel('Time')
        plt.ylabel('Number of Detections')
        plt.title('Object Detections Over Time')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis dates
        plt.gcf().autofmt_xdate()
        
        # Save figure
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
        plt.close()


class DetectionFileHandler(FileSystemEventHandler):
    """Watch for new detection files and process them immediately"""
    
    def __init__(self, processor):
        self.processor = processor
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        if event.src_path.endswith('.json'):
            print(f"New detection file: {os.path.basename(event.src_path)}")
            self.processor.process_file(event.src_path)


def monitor_directory(processor, interval=1.0):
    """Monitor the input directory for new detection files"""
    print(f"Monitoring directory: {processor.input_dir}")
    print("Press Ctrl+C to stop")
    
    event_handler = DetectionFileHandler(processor)
    observer = Observer()
    observer.schedule(event_handler, processor.input_dir, recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(interval)
    except KeyboardInterrupt:
        observer.stop()
    
    observer.join()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Process YOLOv8 detection JSON files')
    parser.add_argument('--input', type=str, default='detections',
                        help='Input directory containing detection JSON files')
    parser.add_argument('--archive', type=str, default=None,
                        help='Archive directory for processed files')
    parser.add_argument('--stats', type=str, default='detection_stats.json',
                        help='Output file for detection statistics')
    parser.add_argument('--plot', type=str, default='detection_trends.png',
                        help='Output file for detection trend plot')
    parser.add_argument('--monitor', action='store_true',
                        help='Monitor input directory for new files')
    parser.add_argument('--delete', action='store_true',
                        help='Delete files after processing (if not archiving)')
    args = parser.parse_args()
    
    # Initialize processor
    processor = DetectionProcessor(args.input, args.archive)
    
    # Process existing files
    processor.process_directory(delete_processed=args.delete)
    
    # Generate and save statistics
    stats = processor.generate_statistics()
    with open(args.stats, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to {args.stats}")
    
    # Generate plot
    processor.plot_detection_trends(args.plot)
    
    # Monitor for new files if requested
    if args.monitor:
        monitor_directory(processor)


if __name__ == "__main__":
    main()