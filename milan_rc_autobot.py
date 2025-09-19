#!/usr/bin/env python3
"""
Robot Control - Web interface for controlling a robot with LiDAR-based obstacle detection and autonomous navigation.

This script provides a web server to control a robot using GPIO for motor and servo control, a LiDAR interface for
obstacle detection, and a machine learning model for autonomous navigation. It includes a WebSocket for real-time control
and a Plotly-based visualization for LiDAR data.

Author: Dragan Bojovic, bojovicd@proton.me
Version: 1.0.0
"""

import asyncio
import csv
import json
import os
import threading
import time
from typing import Dict, List

import numpy as np
import pigpio
from aiohttp import web
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from simple_lidar import SimpleLidarInterface
import pickle

# Configuration constants
MOTOR_SIGNAL = 18
LEVEL_EN = 23
SERVO_PIN = 17
ESC_NEUTRAL = 1500
ESC_FORWARD = 1605
ESC_BACKWARD = 1310
SERVO_LEFT = 1900
SERVO_CENTER = 1500
SERVO_RIGHT = 1100
LIDAR_PORT = '/dev/serial0'
LIDAR_BAUDRATE = 230400
ANGLE_OFFSET = 0
LIDAR_TO_FRONT = 0.24
LIDAR_TO_REAR = 0.16
SAFETY_DISTANCE_FRONT = 0.15
SAFETY_DISTANCE_REAR = 0.15
COLLISION_THRESHOLD = LIDAR_TO_FRONT + SAFETY_DISTANCE_FRONT
COLLISION_THRESHOLD_REAR = LIDAR_TO_REAR + SAFETY_DISTANCE_REAR
FRONT_ANGLE_RANGE = 40
REAR_ANGLE_RANGE = 50
FRONT_ANGLE_MIN = (360 - FRONT_ANGLE_RANGE // 2) % 360
FRONT_ANGLE_MAX = FRONT_ANGLE_RANGE // 2
REAR_ANGLE_MIN = 180 - REAR_ANGLE_RANGE // 2
REAR_ANGLE_MAX = 180 + REAR_ANGLE_RANGE // 2

# Global state
pi = None
current_servo_pulse = SERVO_CENTER
lidar = None
lidar_running = False
safety_running = False
current_direction = None
collision_front = False
collision_rear = False
last_front_distance = None
last_rear_distance = None
collect_mode = False
data = []
autonomous_running = False
model = None
stop_lock = threading.Lock()

def get_features(point_cloud: List[Dict]) -> List[float]:
    """Extract features from LiDAR point cloud for machine learning."""
    angles = np.array([p['angle'] for p in point_cloud])
    distances = np.array([p['distance'] for p in point_cloud])
    adjusted_angles = (angles + ANGLE_OFFSET) % 360
    sectors = np.minimum((adjusted_angles // 30).astype(int), 11)
    features = np.full(12, 2.0)
    for i in range(12):
        mask = sectors == i
        if np.any(mask):
            features[i] = np.min(distances[mask])
    return features.tolist()

def determine_label() -> str:
    """Determine the current action label based on direction and steering."""
    if current_direction is None:
        return 'stop'
    label = current_direction
    if current_servo_pulse == SERVO_LEFT:
        label += '_left'
    elif current_servo_pulse == SERVO_RIGHT:
        label += '_right'
    else:
        label += '_straight'
    return label

def execute_action(label: str) -> None:
    """Execute the specified action (e.g., move forward, turn left)."""
    parts = label.split('_')
    direction = parts[0]
    steering = parts[1] if len(parts) > 1 else 'straight'
    if direction == 'stop':
        stop()
    elif direction == 'forward':
        forward()
    elif direction == 'backward':
        backward_sync()
    else:
        stop()
    if steering == 'left':
        left()
    elif steering == 'right':
        right()
    else:
        center()

def emergency_stop() -> None:
    """Stop the robot immediately due to obstacle detection."""
    global current_direction
    with stop_lock:
        current_direction = None
        pi.set_servo_pulsewidth(MOTOR_SIGNAL, ESC_NEUTRAL)
        pi.set_servo_pulsewidth(SERVO_PIN, SERVO_CENTER)

def safety_monitor() -> None:
    """Monitor LiDAR data for obstacles and trigger emergency stop if needed."""
    global collision_front, collision_rear, current_direction
    while safety_running:
        if lidar and lidar.is_connected:
            point_cloud = lidar.get_point_cloud()
            if point_cloud:
                adjusted_angle = lambda angle: (angle + ANGLE_OFFSET) % 360
                front_points = [p for p in point_cloud if
                                (adjusted_angle(p['angle']) >= FRONT_ANGLE_MIN or
                                 adjusted_angle(p['angle']) <= FRONT_ANGLE_MAX)]
                rear_points = [p for p in point_cloud if
                               REAR_ANGLE_MIN <= adjusted_angle(p['angle']) <= REAR_ANGLE_MAX]
                collision_front = len([p for p in front_points if p['distance'] < COLLISION_THRESHOLD]) >= 2
                collision_rear = len([p for p in rear_points if p['distance'] < COLLISION_THRESHOLD_REAR]) >= 2
                if collision_front and current_direction == 'forward':
                    emergency_stop()
                elif collision_rear and current_direction == 'backward':
                    emergency_stop()
                if collect_mode:
                    label = determine_label()
                    features = get_features(point_cloud)
                    data.append(features + [label])
        time.sleep(0.2)

def lidar_worker() -> None:
    """Continuously read and parse LiDAR data."""
    global lidar_running
    while lidar_running:
        if lidar and lidar.is_connected:
            lidar.read_and_parse_data()
        time.sleep(0.1)

def autonomous_loop() -> None:
    """Run the autonomous navigation loop using the trained model."""
    global autonomous_running, lidar, model
    while autonomous_running:
        if lidar and lidar.is_connected and model:
            point_cloud = lidar.get_point_cloud()
            if point_cloud:
                features = get_features(point_cloud)
                label = model.predict([features])[0]
                execute_action(label)
        time.sleep(0.2)

def setup_gpio() -> None:
    """Initialize GPIO pins for motor and servo control."""
    global pi
    pi = pigpio.pi()
    if not pi.connected:
        raise RuntimeError("pigpio connection failed")
    pi.set_mode(LEVEL_EN, pigpio.OUTPUT)
    pi.write(LEVEL_EN, 1)
    pi.set_PWM_frequency(MOTOR_SIGNAL, 50)
    pi.set_PWM_frequency(SERVO_PIN, 50)
    pi.set_servo_pulsewidth(MOTOR_SIGNAL, ESC_NEUTRAL)
    pi.set_servo_pulsewidth(SERVO_PIN, SERVO_CENTER)

def setup_lidar() -> bool:
    """Initialize the LiDAR interface."""
    global lidar, lidar_running, safety_running
    try:
        lidar = SimpleLidarInterface(port=LIDAR_PORT, baudrate=LIDAR_BAUDRATE)
        if lidar.connect():
            lidar_running = True
            threading.Thread(target=lidar_worker, daemon=True).start()
            safety_running = True
            threading.Thread(target=safety_monitor, daemon=True).start()
            return True
        return False
    except Exception as e:
        print(f"LiDAR setup error: {e}")
        return False

async def index(request: web.Request) -> web.Response:
    """Serve the main HTML interface for robot control."""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Robot Control</title>
        <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
                margin: 0;
                padding: 10px;
                background: #f0f0f0;
                height: 100vh;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }
            h1 { font-size: 24px; margin: 10px 0; }
            p { font-size: 16px; margin: 5px 0; }
            #status { font-size: 16px; font-weight: bold; color: #333; margin: 10px 0; }
            .lidar-info {
                background: #fff;
                border: 2px solid #ddd;
                border-radius: 8px;
                padding: 10px;
                margin: 10px 0;
                font-size: 14px;
            }
            .indicator { display: inline-block; margin: 0 15px; }
            .dot {
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                margin-right: 5px;
            }
            .controls {
                margin: 10px 0;
                display: flex;
                justify-content: center;
                gap: 10px;
                flex-wrap: wrap;
            }
            .controls label {
                font-size: 14px;
                margin-right: 5px;
            }
            .controls input {
                width: 80px;
                padding: 5px;
                font-size: 14px;
            }
            #plot { width: 400px; height: 400px; margin: 0 auto; }
            #toggle-safety-button {
                padding: 5px 10px;
                font-size: 14px;
                border-radius: 5px;
                cursor: pointer;
            }
            #toggle-safety-button.enabled {
                background-color: #00FF00;
            }
            #toggle-safety-button.disabled {
                background-color: #FF0000;
            }
        </style>
        <script>
            const zoomLevels = [[0, 3], [0, 6], [0, 12], [0, 24]];
            let currentZoomIndex = 2; // Start at [0, 12]

            async function sendCommand(cmd) {
                try {
                    const response = await fetch(`/${cmd}`);
                    const text = await response.text();
                    document.getElementById('status').innerText = `Status: ${text}`;
                    if (cmd === 'toggle_safety') {
                        updateSafetyButton();
                    }
                } catch (error) {
                    document.getElementById('status').innerText = `Error: ${error}`;
                }
            }

            async function updateLidarStatus() {
                try {
                    const response = await fetch('/lidar_status');
                    const status = await response.json();
                    document.getElementById('front-dot').style.backgroundColor = status.collision_front ? '#FF0000' : '#00FF00';
                    document.getElementById('front-text').textContent = status.collision_front ? 'Obstacle!' : 'Clear';
                    document.getElementById('front-distance').textContent = status.front_distance !== null ? `${status.front_distance.toFixed(2)}m` : '0.00m';
                    document.getElementById('rear-dot').style.backgroundColor = status.collision_rear ? '#FF0000' : '#00FF00';
                    document.getElementById('rear-text').textContent = status.collision_rear ? 'Obstacle!' : 'Clear';
                    document.getElementById('rear-distance').textContent = status.rear_distance !== null ? `${status.rear_distance.toFixed(2)}m` : '0.00m';
                    document.getElementById('angle-range-display').textContent = `Front: ${status.front_angle_min}°-${status.front_angle_max}°, Rear: ${status.rear_angle_min}°-${status.rear_angle_max}°, Front Safety: ${status.safety_distance_front}m, Rear Safety: ${status.safety_distance_rear}m, ESC Forward: ${status.esc_forward}, ESC Backward: ${status.esc_backward}, LiDAR: ${status.lidar_running ? 'Running' : 'Stopped'}, Safety: ${status.safety_running ? 'Running' : 'Stopped'}`;
                    const safetyButton = document.getElementById('toggle-safety-button');
                    safetyButton.className = status.safety_running ? 'enabled' : 'disabled';
                    safetyButton.textContent = status.safety_running ? 'Disable Safety' : 'Enable Safety';
                } catch (error) {
                    document.getElementById('front-text').textContent = 'Error';
                    document.getElementById('front-distance').textContent = '0.00m';
                    document.getElementById('rear-text').textContent = 'Error';
                    document.getElementById('rear-distance').textContent = '0.00m';
                }
            }

            async function updateSafetyButton() {
                try {
                    const response = await fetch('/lidar_status');
                    const status = await response.json();
                    const safetyButton = document.getElementById('toggle-safety-button');
                    safetyButton.className = status.safety_running ? 'enabled' : 'disabled';
                    safetyButton.textContent = status.safety_running ? 'Disable Safety' : 'Enable Safety';
                } catch (error) {
                    console.error('Safety button update error:', error);
                }
            }

            async function updatePlot() {
                try {
                    const response = await fetch('/lidar_data');
                    const data = await response.json();
                    if (!data.angles.length || !data.distances.length) {
                        Plotly.react('plot', [], {
                            polar: { radialaxis: { visible: true, range: zoomLevels[currentZoomIndex] } },
                            showlegend: false,
                            margin: { t: 20, b: 20, l: 20, r: 20 }
                        });
                        return;
                    }
                    Plotly.react('plot', [{
                        type: 'scatterpolar',
                        r: data.distances,
                        theta: data.angles,
                        mode: 'markers',
                        marker: { color: 'blue', size: 5 }
                    }], {
                        polar: {
                            radialaxis: { visible: true, range: zoomLevels[currentZoomIndex] },
                            angularaxis: { direction: 'clockwise' }
                        },
                        showlegend: false,
                        dragmode: 'zoom',
                        margin: { t: 20, b: 20, l: 20, r: 20 },
                        uirevision: 'true',
                        updatemenus: [{
                            buttons: [
                                {
                                    method: 'relayout',
                                    args: ['polar.radialaxis.range', [0, 12]],
                                    label: 'Reset Zoom'
                                }
                            ],
                            direction: 'left',
                            pad: {'r': 10, 't': 10},
                            showactive: true,
                            type: 'buttons',
                            x: 0.1,
                            xanchor: 'left',
                            y: 1.1,
                            yanchor: 'top'
                        }]
                    }, {
                        responsive: true,
                        scrollZoom: true
                    });
                } catch (error) {
                    document.getElementById('status').innerText = `Plot error: ${error}`;
                }
            }

            async function updateSettings() {
                const safetyDistanceFront = document.getElementById('safety-distance-front').value;
                const safetyDistanceRear = document.getElementById('safety-distance-rear').value;
                const frontAngleRange = document.getElementById('front-angle-range').value;
                const rearAngleRange = document.getElementById('rear-angle-range').value;
                const escForward = document.getElementById('esc-forward').value;
                const escBackward = document.getElementById('esc-backward').value;
                try {
                    const response = await fetch('/set_settings', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            safety_distance_front: parseFloat(safetyDistanceFront),
                            safety_distance_rear: parseFloat(safetyDistanceRear),
                            front_angle_range: parseInt(frontAngleRange),
                            rear_angle_range: parseInt(rearAngleRange),
                            esc_forward: parseInt(escForward),
                            esc_backward: parseInt(escBackward)
                        })
                    });
                    document.getElementById('status').innerText = `Status: ${await response.text()}`;
                } catch (error) {
                    document.getElementById('status').innerText = `Error: ${error}`;
                }
            }

            const ws = new WebSocket('ws://' + location.hostname + ':8080/ws');
            ws.onopen = () => console.log('WebSocket connected');
            ws.onmessage = (event) => document.getElementById('status').innerText = `Status: ${event.data}`;
            ws.onclose = () => console.log('WebSocket closed');

            document.addEventListener('keydown', (event) => {
                if (event.repeat) return;
                const commands = {
                    'ArrowUp': 'forward',
                    'ArrowDown': 'backward',
                    'ArrowLeft': 'start_left',
                    'ArrowRight': 'start_right',
                    ' ': 'stop'
                };
                if (commands[event.key]) {
                    ws.send(commands[event.key]);
                }
            });

            document.addEventListener('keyup', (event) => {
                const commands = {
                    'ArrowUp': 'stop',
                    'ArrowDown': 'stop',
                    'ArrowLeft': 'release_steering',
                    'ArrowRight': 'release_steering'
                };
                if (commands[event.key]) {
                    ws.send(commands[event.key]);
                }
            });

            window.onload = () => {
                fetch('/lidar_status')
                    .then(response => response.json())
                    .then(status => {
                        document.getElementById('safety-distance-front').value = status.safety_distance_front;
                        document.getElementById('safety-distance-rear').value = status.safety_distance_rear;
                        document.getElementById('front-angle-range').value = status.front_angle_range;
                        document.getElementById('rear-angle-range').value = status.rear_angle_range;
                        document.getElementById('esc-forward').value = status.esc_forward;
                        document.getElementById('esc-backward').value = status.esc_backward;
                        updateLidarStatus();
                    })
                    .catch(error => {
                        document.getElementById('status').innerText = `Error loading initial settings: ${error}`;
                    });
                setInterval(updateLidarStatus, 100);
                Plotly.newPlot('plot', [], {
                    polar: { radialaxis: { visible: true, range: [0, 12] } },
                    dragmode: 'zoom',
                    margin: { t: 20, b: 20, l: 20, r: 20 }
                });
                setInterval(updatePlot, 200);

                const plotDiv = document.getElementById('plot');
                plotDiv.onwheel = (event) => {
                    event.preventDefault();
                    const delta = event.deltaY > 0 ? 1 : -1;
                    currentZoomIndex = Math.max(0, Math.min(zoomLevels.length - 1, currentZoomIndex + delta));
                    Plotly.relayout('plot', {
                        'polar.radialaxis.range': zoomLevels[currentZoomIndex]
                    });
                };
            };
        </script>
    </head>
    <body>
        <h1>Robot Control</h1>
        <p>Use arrow keys to move (Up/Down) and steer (Left/Right). Space to stop.</p>
        <div id="status">Status: Ready</div>
        <div class="lidar-info">
            <div class="indicator">
                <span class="dot" id="front-dot"></span>Front: <span id="front-text">Checking...</span>, Distance: <span id="front-distance">0.00m</span>
            </div>
            <div class="indicator">
                <span class="dot" id="rear-dot"></span>Rear: <span id="rear-text">Checking...</span>, Distance: <span id="rear-distance">0.00m</span>
            </div>
            <div>Current Settings: <span id="angle-range-display"></span></div>
        </div>
        <div id="plot"></div>
        <div class="controls">
            <div>
                <label>Front Safety Distance (m):</label>
                <input type="number" id="safety-distance-front" min="0.1" max="2.0" step="0.1" value="0.50">
            </div>
            <div>
                <label>Rear Safety Distance (m):</label>
                <input type="number" id="safety-distance-rear" min="0.1" max="2.0" step="0.1" value="0.50">
            </div>
            <div>
                <label>Front Angle Range (°):</label>
                <input type="number" id="front-angle-range" min="10" max="180" step="10" value="25">
            </div>
            <div>
                <label>Rear Angle Range (°):</label>
                <input type="number" id="rear-angle-range" min="10" max="180" step="10" value="50">
            </div>
            <div>
                <label>ESC Forward (µs):</label>
                <input type="number" id="esc-forward" min="1500" max="2000" step="10" value="1630">
            </div>
            <div>
                <label>ESC Backward (µs):</label>
                <input type="number" id="esc-backward" min="1000" max="1500" step="10" value="1270">
            </div>
            <button onclick="updateSettings()">Update Settings</button>
            <button onclick="sendCommand('start_collect')">Start Collect</button>
            <button onclick="sendCommand('stop_collect')">Stop Collect</button>
            <button onclick="sendCommand('save_data')">Save Data</button>
            <button onclick="sendCommand('train_model')">Train Model</button>
            <button onclick="sendCommand('start_autonomous')">Start Autonomous</button>
            <button onclick="sendCommand('stop_autonomous')">Stop Autonomous</button>
            <button id="toggle-safety-button" onclick="sendCommand('toggle_safety')">Toggle Safety</button>
        </div>
    </body>
    </html>
    """
    return web.Response(text=html, content_type='text/html')

def forward() -> str:
    """Move the robot forward, checking for obstacles."""
    global current_direction
    if safety_running and collision_front:
        emergency_stop()
        return "Forward Blocked!"
    current_direction = 'forward'
    pi.set_servo_pulsewidth(MOTOR_SIGNAL, ESC_FORWARD)
    return "Forward"

def backward_sync() -> str:
    """Move the robot backward with a brief neutral pause."""
    global current_direction
    if safety_running and collision_rear:
        emergency_stop()
        return "Backward Blocked!"
    current_direction = 'backward'
    pi.set_servo_pulsewidth(MOTOR_SIGNAL, ESC_BACKWARD)
    time.sleep(0.1)
    pi.set_servo_pulsewidth(MOTOR_SIGNAL, ESC_NEUTRAL)
    time.sleep(0.2)
    pi.set_servo_pulsewidth(MOTOR_SIGNAL, ESC_BACKWARD)
    return "Backward"

async def backward() -> str:
    """Asynchronous version of backward movement."""
    global current_direction
    if safety_running and collision_rear:
        emergency_stop()
        return "Backward Blocked!"
    current_direction = 'backward'
    pi.set_servo_pulsewidth(MOTOR_SIGNAL, ESC_BACKWARD)
    await asyncio.sleep(0.1)
    pi.set_servo_pulsewidth(MOTOR_SIGNAL, ESC_NEUTRAL)
    await asyncio.sleep(0.2)
    pi.set_servo_pulsewidth(MOTOR_SIGNAL, ESC_BACKWARD)
    return "Backward"

def stop() -> str:
    """Stop the robot and reset steering."""
    global current_direction
    with stop_lock:
        current_direction = None
        pi.set_servo_pulsewidth(MOTOR_SIGNAL, ESC_NEUTRAL)
        pi.set_servo_pulsewidth(SERVO_PIN, SERVO_CENTER)
    return "Stop"

def left() -> str:
    """Steer the robot left."""
    global current_servo_pulse
    pi.set_servo_pulsewidth(SERVO_PIN, SERVO_LEFT)
    current_servo_pulse = SERVO_LEFT
    return "Left"

def right() -> str:
    """Steer the robot right."""
    global current_servo_pulse
    pi.set_servo_pulsewidth(SERVO_PIN, SERVO_RIGHT)
    current_servo_pulse = SERVO_RIGHT
    return "Right"

def center() -> str:
    """Center the robot's steering."""
    global current_servo_pulse
    pi.set_servo_pulsewidth(SERVO_PIN, SERVO_CENTER)
    current_servo_pulse = SERVO_CENTER
    return "Center"

async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """Handle WebSocket commands for real-time robot control."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    async for msg in ws:
        if msg.type == web.WSMsgType.TEXT:
            cmd = msg.data
            if cmd == 'forward':
                await ws.send_str(forward())
            elif cmd == 'backward':
                await ws.send_str(await backward())
            elif cmd == 'stop':
                await ws.send_str(stop())
            elif cmd == 'start_left':
                await ws.send_str(left())
            elif cmd == 'start_right':
                await ws.send_str(right())
            elif cmd == 'release_steering':
                await ws.send_str(center())
            elif cmd == 'toggle_safety':
                await ws.send_str((await toggle_safety_async()).text)
    return ws

async def toggle_safety_async(request: web.Request = None) -> web.Response:
    """Toggle the safety monitor on or off."""
    global safety_running
    safety_running = not safety_running
    if safety_running:
        threading.Thread(target=safety_monitor, daemon=True).start()
        return web.Response(text="Safety Enabled")
    return web.Response(text="Safety Disabled")

async def start_collect_async(request: web.Request) -> web.Response:
    """Start collecting training data."""
    global collect_mode, data
    collect_mode = True
    data = []
    return web.Response(text="Collecting")

async def stop_collect_async(request: web.Request) -> web.Response:
    """Stop collecting training data."""
    global collect_mode
    collect_mode = False
    return web.Response(text="Stopped collecting")

async def save_data_async(request: web.Request) -> web.Response:
    """Save collected training data to a CSV file."""
    global data
    if data:
        with open('data.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f'feature{i}' for i in range(12)] + ['label'])
            writer.writerows(data)
        data = []
        return web.Response(text="Data saved")
    return web.Response(text="No data")

async def train_model_async(request: web.Request) -> web.Response:
    """Train or load a decision tree model for autonomous navigation."""
    global model
    if os.path.exists('model.pkl'):
        try:
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            return web.Response(text="Model loaded")
        except Exception:
            return web.Response(text="Error loading model", status=400)
    elif os.path.exists('data.csv'):
        data_np = np.genfromtxt('data.csv', delimiter=',', skip_header=1, dtype=str)
        if data_np.size == 0:
            return web.Response(text="No valid data")
        X = data_np[:, :-1].astype(float)
        y = data_np[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        return web.Response(text=f"Trained, accuracy {accuracy:.2f}")
    return web.Response(text="No data file or model")

async def start_autonomous_async(request: web.Request) -> web.Response:
    """Start autonomous navigation mode."""
    global autonomous_running
    if model:
        if not autonomous_running:
            autonomous_running = True
            threading.Thread(target=autonomous_loop, daemon=True).start()
        return web.Response(text="Autonomous started")
    return web.Response(text="No model")

async def stop_autonomous_async(request: web.Request) -> web.Response:
    """Stop autonomous navigation mode."""
    global autonomous_running
    autonomous_running = False
    stop()
    return web.Response(text="Autonomous stopped")

async def set_settings(request: web.Request) -> web.Response:
    """Update configuration settings from the web interface."""
    global SAFETY_DISTANCE_FRONT, SAFETY_DISTANCE_REAR, COLLISION_THRESHOLD, COLLISION_THRESHOLD_REAR
    global FRONT_ANGLE_RANGE, REAR_ANGLE_RANGE, FRONT_ANGLE_MIN, FRONT_ANGLE_MAX, REAR_ANGLE_MIN, REAR_ANGLE_MAX
    global ESC_FORWARD, ESC_BACKWARD
    try:
        data = await request.json()
        safety_distance_front = float(data.get('safety_distance_front', SAFETY_DISTANCE_FRONT))
        safety_distance_rear = float(data.get('safety_distance_rear', SAFETY_DISTANCE_REAR))
        front_angle_range = int(data.get('front_angle_range', FRONT_ANGLE_RANGE))
        rear_angle_range = int(data.get('rear_angle_range', REAR_ANGLE_RANGE))
        esc_forward = int(data.get('esc_forward', ESC_FORWARD))
        esc_backward = int(data.get('esc_backward', ESC_BACKWARD))
        if not (0.1 <= safety_distance_front <= 2.0) or not (0.1 <= safety_distance_rear <= 2.0):
            return web.Response(text="Safety distances must be between 0.1 and 2.0 meters.", status=400)
        if not (10 <= front_angle_range <= 180) or not (10 <= rear_angle_range <= 180):
            return web.Response(text="Angle ranges must be between 10 and 180 degrees.", status=400)
        if not (1500 <= esc_forward <= 2000) or not (1000 <= esc_backward <= 1500):
            return web.Response(text="ESC Forward must be between 1500 and 2000 µs, ESC Backward between 1000 and 1500 µs.", status=400)
        SAFETY_DISTANCE_FRONT = safety_distance_front
        SAFETY_DISTANCE_REAR = safety_distance_rear
        COLLISION_THRESHOLD = LIDAR_TO_FRONT + SAFETY_DISTANCE_FRONT
        COLLISION_THRESHOLD_REAR = LIDAR_TO_REAR + SAFETY_DISTANCE_REAR
        FRONT_ANGLE_RANGE = front_angle_range
        REAR_ANGLE_RANGE = rear_angle_range
        FRONT_ANGLE_MIN = (360 - FRONT_ANGLE_RANGE // 2) % 360
        FRONT_ANGLE_MAX = FRONT_ANGLE_RANGE // 2
        REAR_ANGLE_MIN = 180 - REAR_ANGLE_RANGE // 2
        REAR_ANGLE_MAX = 180 + REAR_ANGLE_RANGE // 2
        ESC_FORWARD = esc_forward
        ESC_BACKWARD = esc_backward
        return web.json_response({
            "message": "Settings updated",
            "safety_distance_front": SAFETY_DISTANCE_FRONT,
            "safety_distance_rear": SAFETY_DISTANCE_REAR,
            "front_angle_range": FRONT_ANGLE_RANGE,
            "rear_angle_range": REAR_ANGLE_RANGE,
            "esc_forward": ESC_FORWARD,
            "esc_backward": ESC_BACKWARD
        })
    except Exception as e:
        return web.Response(text=f"Error: {e}", status=400)

async def get_lidar_status(request: web.Request) -> web.Response:
    """Return the current LiDAR and safety status."""
    global collision_front, collision_rear, last_front_distance, last_rear_distance, lidar_running, safety_running
    status = {
        "connected": False,
        "collision_front": collision_front,
        "collision_rear": collision_rear,
        "front_distance": last_front_distance,
        "rear_distance": last_rear_distance,
        "front_angle_min": FRONT_ANGLE_MIN,
        "front_angle_max": FRONT_ANGLE_MAX,
        "rear_angle_min": REAR_ANGLE_MIN,
        "rear_angle_max": REAR_ANGLE_MAX,
        "safety_distance_front": SAFETY_DISTANCE_FRONT,
        "safety_distance_rear": SAFETY_DISTANCE_REAR,
        "front_angle_range": FRONT_ANGLE_RANGE,
        "rear_angle_range": REAR_ANGLE_RANGE,
        "esc_forward": ESC_FORWARD,
        "esc_backward": ESC_BACKWARD,
        "lidar_running": lidar_running,
        "safety_running": safety_running
    }
    if lidar and lidar.is_connected:
        point_cloud = lidar.get_point_cloud()
        if point_cloud:
            adjusted_angle = lambda angle: (angle + ANGLE_OFFSET) % 360
            front_points_distance = [p for p in point_cloud if
                                    (adjusted_angle(p['angle']) >= FRONT_ANGLE_MIN or
                                     adjusted_angle(p['angle']) <= FRONT_ANGLE_MAX)]
            rear_points_distance = [p for p in point_cloud if
                                   REAR_ANGLE_MIN <= adjusted_angle(p['angle']) <= REAR_ANGLE_MAX]
            if front_points_distance:
                last_front_distance = min([p['distance'] for p in front_points_distance])
            if rear_points_distance:
                last_rear_distance = min([p['distance'] for p in rear_points_distance])
            status["front_distance"] = last_front_distance
            status["rear_distance"] = last_rear_distance
            status["connected"] = True
            status.update(lidar.get_status())
    return web.json_response(status)

async def get_lidar_data(request: web.Request) -> web.Response:
    """Return the current LiDAR point cloud data."""
    if lidar and lidar.is_connected:
        point_cloud = lidar.get_point_cloud()
        if point_cloud:
            angles = [p['angle'] for p in point_cloud]
            distances = [p['distance'] for p in point_cloud]
            return web.json_response({'angles': angles, 'distances': distances})
        return web.json_response({'angles': [], 'distances': []})
    return web.json_response({'angles': [], 'distances': []})

def cleanup() -> None:
    """Clean up resources before shutting down."""
    global lidar_running, safety_running, lidar, pi
    try:
        if safety_running:
            safety_running = False
            time.sleep(1.0)
        if lidar_running:
            lidar_running = False
            time.sleep(2.0)
        if lidar:
            lidar.disconnect()
        if pi:
            pi.set_servo_pulsewidth(MOTOR_SIGNAL, ESC_NEUTRAL)
            pi.set_servo_pulsewidth(SERVO_PIN, SERVO_CENTER)
            time.sleep(0.1)
            pi.set_servo_pulsewidth(MOTOR_SIGNAL, 0)
            pi.set_servo_pulsewidth(SERVO_PIN, 0)
            pi.stop()
    except Exception as e:
        print(f"Cleanup error: {e}")

app = web.Application()
app.add_routes([
    web.get('/', index),
    web.get('/ws', websocket_handler),
    web.get('/lidar_status', get_lidar_status),
    web.get('/lidar_data', get_lidar_data),
    web.post('/set_settings', set_settings),
    web.get('/start_collect', start_collect_async),
    web.get('/stop_collect', stop_collect_async),
    web.get('/save_data', save_data_async),
    web.get('/train_model', train_model_async),
    web.get('/start_autonomous', start_autonomous_async),
    web.get('/stop_autonomous', stop_autonomous_async),
    web.get('/toggle_safety', toggle_safety_async)
])

if __name__ == '__main__':
    try:
        setup_gpio()
        setup_lidar()
        loop = asyncio.get_event_loop()
        web.run_app(app, host='0.0.0.0', port=8080)
    except KeyboardInterrupt:
        print("Server interrupted")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        cleanup()
        loop = asyncio.get_event_loop()
        tasks = [task for task in asyncio.all_tasks(loop) if task is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()