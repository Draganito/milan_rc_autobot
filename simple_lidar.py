#!/usr/bin/env python3
"""
Simple LiDAR Interface - Reusable interface for LiDAR sensors.

This class provides a clean API for handling LiDAR data, optimized for performance
and compatibility with LD19/LD06 and experimental support for D500 sensors.

Author: Dragan Bojovic, bojovicd@proton.me
Version: 1.0.0
"""

import math
import serial
import threading
import time
from typing import List, Dict, Any

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("NumPy not available - install with: pip install numpy")

class SimpleLidarInterface:
    """Modular interface for LiDAR sensors, optimized for Raspberry Pi."""
    
    def __init__(self, port: str = '/dev/serial0', baudrate: int = 230400, buffer_size: int = 1000, timeout: float = 1.0):
        """
        Initialize the LiDAR interface.

        Args:
            port: Serial port (e.g., '/dev/serial0', 'COM1').
            baudrate: Baudrate for the LiDAR (230400 for D500/LD19).
            buffer_size: Maximum number of points in the buffer.
            timeout: Timeout for serial connection.
        """
        self.port = port
        self.baudrate = baudrate
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.serial = None
        self.is_connected = False
        self.data_buffer = bytearray()
        self.points_buffer = []
        self.lock = threading.Lock()
        
        # CRC table for checksum calculation
        self.crc_table = [
            0x00, 0x4d, 0x9a, 0xd7, 0x79, 0x34, 0xe3, 0xae, 0xf2, 0xbf, 0x68, 0x25, 0x8b, 0xc6, 0x11, 0x5c,
            0xa9, 0xe4, 0x33, 0x7e, 0xd0, 0x9d, 0x4a, 0x07, 0x5b, 0x16, 0xc1, 0x8c, 0x22, 0x6f, 0xb8, 0xf5,
            0x1f, 0x52, 0x85, 0xc8, 0x66, 0x2b, 0xfc, 0xb1, 0xed, 0xa0, 0x77, 0x3a, 0x94, 0xd9, 0x0e, 0x43,
            0xb6, 0xfb, 0x2c, 0x61, 0xcf, 0x82, 0x55, 0x18, 0x44, 0x09, 0xde, 0x93, 0x3d, 0x70, 0xa7, 0xea,
            0x3e, 0x73, 0xa4, 0xe9, 0x47, 0x0a, 0xdd, 0x90, 0xcc, 0x81, 0x56, 0x1b, 0xb5, 0xf8, 0x2f, 0x62,
            0x97, 0xda, 0x0d, 0x40, 0xee, 0xa3, 0x74, 0x39, 0x65, 0x28, 0xff, 0xb2, 0x1c, 0x51, 0x86, 0xcb,
            0x21, 0x6c, 0xbb, 0xf6, 0x58, 0x15, 0xc2, 0x8f, 0xd3, 0x9e, 0x49, 0x04, 0xaa, 0xe7, 0x30, 0x7d,
            0x88, 0xc5, 0x12, 0x5f, 0xf1, 0xbc, 0x6b, 0x26, 0x7a, 0x37, 0xe0, 0xad, 0x03, 0x4e, 0x99, 0xd4,
            0x7c, 0x31, 0xe6, 0xab, 0x05, 0x48, 0x9f, 0xd2, 0x8e, 0xc3, 0x14, 0x59, 0xf7, 0xba, 0x6d, 0x20,
            0xd5, 0x98, 0x4f, 0x02, 0xac, 0xe1, 0x36, 0x7b, 0x27, 0x6a, 0xbd, 0xf0, 0x5e, 0x13, 0xc4, 0x89,
            0x63, 0x2e, 0xf9, 0xb4, 0x1a, 0x57, 0x80, 0xcd, 0x91, 0xdc, 0x0b, 0x46, 0xe8, 0xa5, 0x72, 0x3f,
            0xca, 0x87, 0x50, 0x1d, 0xb3, 0xfe, 0x29, 0x64, 0x38, 0x75, 0xa2, 0xef, 0x41, 0x0c, 0xdb, 0x96,
            0x42, 0x0f, 0xd8, 0x95, 0x3b, 0x76, 0xa1, 0xec, 0xb0, 0xfd, 0x2a, 0x67, 0xc9, 0x84, 0x53, 0x1e,
            0xeb, 0xa6, 0x71, 0x3c, 0x92, 0xdf, 0x08, 0x45, 0x19, 0x54, 0x83, 0xce, 0x60, 0x2d, 0xfa, 0xb7,
            0x5d, 0x10, 0xc7, 0x8a, 0x24, 0x69, 0xbe, 0xf3, 0xaf, 0xe2, 0x35, 0x78, 0xd6, 0x9b, 0x4c, 0x01,
            0xf4, 0xb9, 0x6e, 0x23, 0x8d, 0xc0, 0x17, 0x5a, 0x06, 0x4b, 0x9c, 0xd1, 0x7f, 0x32, 0xe5, 0xa8
        ]

    def connect(self) -> bool:
        """Establish a connection to the LiDAR sensor."""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
            self.is_connected = True
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            self.is_connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from the LiDAR sensor."""
        if self.serial:
            self.serial.close()
        self.is_connected = False

    def calculate_crc(self, data: bytes) -> int:
        """Calculate CRC-8 checksum for a data packet."""
        crc = 0
        for byte in data:
            crc = self.crc_table[(crc ^ byte) & 0xff]
        return crc

    def read_and_parse_data(self) -> List[Dict[str, Any]]:
        """Read and parse LiDAR data into a point cloud."""
        if not self.is_connected or not self.serial:
            return []
        try:
            if self.serial.in_waiting > 0:
                data = self.serial.read(self.serial.in_waiting)
                self.data_buffer.extend(data)
                if len(self.data_buffer) > 100 * 47:
                    self.data_buffer.clear()
                new_points = self._process_data_buffer()
                with self.lock:
                    self.points_buffer.extend(new_points)
                    if len(self.points_buffer) > self.buffer_size:
                        self.points_buffer = self.points_buffer[-self.buffer_size:]
                return new_points
        except Exception as e:
            print(f"Read error: {e}")
        return []

    def _process_data_buffer(self) -> List[Dict[str, Any]]:
        """Process the data buffer and extract point cloud data."""
        points = []
        while len(self.data_buffer) >= 47:
            header_pos = -1
            for i in range(min(10, len(self.data_buffer) - 2)):
                if (self.data_buffer[i] == 0x54 and self.data_buffer[i+1] == 0x2C) or \
                   (self.data_buffer[i] == 0x54 and self.data_buffer[i+1] == 0xAC) or \
                   (self.data_buffer[i] == 0xAA and self.data_buffer[i+1] == 0x55):
                    header_pos = i
                    break
            if header_pos == -1:
                self.data_buffer.pop(0)
                continue
            if header_pos > 0:
                del self.data_buffer[:header_pos]
            if len(self.data_buffer) < 47:
                break
            packet = self.data_buffer[:47]
            calculated_crc = self.calculate_crc(packet[:-1])
            received_crc = packet[-1]
            if calculated_crc == received_crc:
                if packet[0] == 0x54:
                    points.extend(self._parse_ld19_packet(packet))
                elif packet[0] == 0xAA:
                    points.extend(self._parse_d500_packet(packet))
            del self.data_buffer[:47]
        return points

    def _parse_ld19_packet(self, packet: bytes) -> List[Dict[str, Any]]:
        """Parse an LD19 packet into point cloud data."""
        points = []
        try:
            speed = int.from_bytes(packet[2:4], 'little')
            start_angle = int.from_bytes(packet[4:6], 'little') / 100.0
            end_angle = int.from_bytes(packet[42:44], 'little') / 100.0
            end_angle = end_angle % 360.0
            angle_diff = (end_angle - start_angle + 360) % 360
            angle_step = angle_diff / 11 if 11 > 0 else 0
            angles = []
            distances = []
            confidences = []
            for i in range(12):
                offset = 6 + i * 3
                if offset + 3 > len(packet):
                    break
                distance_raw = int.from_bytes(packet[offset:offset+2], 'little')
                distance = distance_raw / 1000.0
                confidence = packet[offset+2]
                angle = start_angle + i * angle_step if i < 11 else end_angle
                angle = angle % 360
                if 0.03 <= distance <= 15.0:
                    angles.append(angle)
                    distances.append(distance)
                    confidences.append(confidence)
            if angles:
                timestamp = time.time()
                if HAS_NUMPY:
                    angles_rad = np.radians(angles)
                    x_coords = np.cos(angles_rad) * distances
                    y_coords = np.sin(angles_rad) * distances
                    for i, (angle, distance, confidence, x, y) in enumerate(zip(angles, distances, confidences, x_coords, y_coords)):
                        points.append({
                            'angle': angle,
                            'distance': distance,
                            'confidence': confidence,
                            'x': x,
                            'y': y,
                            'timestamp': timestamp
                        })
                else:
                    for angle, distance, confidence in zip(angles, distances, confidences):
                        x = distance * math.cos(math.radians(angle))
                        y = distance * math.sin(math.radians(angle))
                        points.append({
                            'angle': angle,
                            'distance': distance,
                            'confidence': confidence,
                            'x': x,
                            'y': y,
                            'timestamp': timestamp
                        })
        except Exception as e:
            print(f"LD19 parse error: {e}")
        return points

    def _parse_d500_packet(self, packet: bytes) -> List[Dict[str, Any]]:
        """Parse a D500 packet (fallback to LD19 parser)."""
        return self._parse_ld19_packet(packet)

    def get_point_cloud(self) -> List[Dict[str, Any]]:
        """Return the current point cloud."""
        with self.lock:
            return self.points_buffer.copy()

    def get_point_count(self) -> int:
        """Return the number of points in the buffer."""
        with self.lock:
            return len(self.points_buffer)

    def clear_buffer(self) -> None:
        """Clear the point cloud buffer."""
        with self.lock:
            self.points_buffer.clear()

    def get_status(self) -> Dict[str, Any]:
        """Return status information about the LiDAR."""
        return {
            'connected': self.is_connected,
            'port': self.port,
            'baudrate': self.baudrate,
            'point_count': self.get_point_count(),
            'buffer_size': len(self.data_buffer)
        }

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

def example_usage():
    """Example usage of the SimpleLidarInterface."""
    lidar = SimpleLidarInterface(port='/dev/serial0', baudrate=230400)
    try:
        if not lidar.connect():
            print("Failed to connect to LiDAR")
            return
        while True:
            new_points = lidar.read_and_parse_data()
            point_cloud = lidar.get_point_cloud()
            if point_cloud:
                print(f"{len(point_cloud)} points available")
                for i, point in enumerate(point_cloud[:3]):
                    print(f"Point {i+1}: Angle={point['angle']:.1f}°, Distance={point['distance']:.3f}m, Confidence={point['confidence']}")
                status = lidar.get_status()
                print(f"Status: {status}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Example stopped")
    finally:
        lidar.disconnect()

if __name__ == "__main__":
    example_usage()