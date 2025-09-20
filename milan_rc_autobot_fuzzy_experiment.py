#!/usr/bin/env python3
"""
Robot Control - Web interface for controlling a robot with LiDAR-based obstacle detection and fuzzy logic autonomous navigation.

This script provides a web server to control a robot using GPIO for motor and servo control, a LiDAR interface for
obstacle detection, and a fuzzy logic system for autonomous navigation. It includes a WebSocket for real-time control
and a Plotly-based visualization for LiDAR data with 8 sectors, lines connecting sector boundaries at min distance with
transparent green filled sectors, without Plotly default angular grid lines, ticks, or legend. The safety monitor uses
Sector 0 (337.5°–22.5°) for front and Sector 4 (157.5°–202.5°) for rear. The robot reverses when an obstacle is reached
and attempts to navigate around it.

Author: Dragan Bojovic, bojovicd@proton.me
Version: 1.1.4
"""

import asyncio
import threading
import time
from typing import Dict, List

import numpy as np
import pigpio
from aiohttp import web
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from simple_lidar import SimpleLidarInterface

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
# Safety monitor uses Sector 0 (337.5°–22.5°) for front and Sector 4 (157.5°–202.5°) for rear
FRONT_SECTOR_MIN = 337.5
FRONT_SECTOR_MAX = 22.5
REAR_SECTOR_MIN = 157.5
REAR_SECTOR_MAX = 202.5

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
autonomous_running = False
fuzzy_system = None
sector_inputs = None
stop_lock = threading.Lock()

def get_features(point_cloud: List[Dict]) -> List[float]:
    """Extract features from LiDAR point cloud for fuzzy logic, using 8 sectors."""
    angles = np.array([p['angle'] for p in point_cloud])
    distances = np.array([p['distance'] for p in point_cloud])
    adjusted_angles = (angles + ANGLE_OFFSET) % 360
    sectors = np.zeros_like(adjusted_angles, dtype=int)
    # Define sector boundaries
    sector_boundaries = [337.5, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5]
    for i in range(8):
        start_angle = sector_boundaries[i]
        end_angle = sector_boundaries[(i + 1) % 8]
        if start_angle > end_angle:  # Handle sector crossing 0°
            mask = (adjusted_angles >= start_angle) | (adjusted_angles < end_angle)
        else:
            mask = (adjusted_angles >= start_angle) & (adjusted_angles < end_angle)
        sectors[mask] = i
    features = np.full(8, 2.0)
    for i in range(8):
        mask = sectors == i
        if np.any(mask):
            features[i] = np.min(distances[mask])
    return features.tolist()

def setup_fuzzy_logic():
    """Initialize the fuzzy logic system for autonomous navigation."""
    global fuzzy_system, sector_inputs
    # Inputs: Minimum distances of 8 sectors (0–2m)
    sector_inputs = [ctrl.Antecedent(np.arange(0, 2.1, 0.1), f'sector_{i}') for i in range(8)]
    # Outputs: Direction (stop, forward, backward), Steering (left, right, straight)
    direction = ctrl.Consequent(np.arange(-1, 1.1, 0.1), 'direction')
    steering = ctrl.Consequent(np.arange(-1, 1.1, 0.1), 'steering')

    # Membership functions aligned with safety thresholds
    for sd in sector_inputs:
        sd['near'] = fuzz.trimf(sd.universe, [0, 0, COLLISION_THRESHOLD])  # 0–0.39 m
        sd['medium'] = fuzz.trimf(sd.universe, [COLLISION_THRESHOLD - 0.1, COLLISION_THRESHOLD, COLLISION_THRESHOLD + 0.3])  # 0.29–0.69 m
        sd['far'] = fuzz.trimf(sd.universe, [COLLISION_THRESHOLD, 2.0, 2.0])  # 0.39–2.0 m
    
    direction['backward'] = fuzz.trimf(direction.universe, [-1, -1, 0])
    direction['stop'] = fuzz.trimf(direction.universe, [-0.2, 0, 0.2])
    direction['forward'] = fuzz.trimf(direction.universe, [0, 1, 1])
    
    steering['left'] = fuzz.trimf(steering.universe, [-1, -1, 0])
    steering['straight'] = fuzz.trimf(steering.universe, [-0.2, 0, 0.2])
    steering['right'] = fuzz.trimf(steering.universe, [0, 1, 1])

    # Fuzzy rules - Optimierte Regeln für Dreipunktwende und Hindernisumfahrung
    rules = [
        # Regel 1: Aktiv nach freien Flächen suchen - vorwärts in Richtung des freiesten Weges
        ctrl.Rule(sector_inputs[0]['far'] & sector_inputs[1]['far'] &
                  (sector_inputs[2]['far'] | sector_inputs[3]['far']) &
                  (sector_inputs[6]['medium'] | sector_inputs[7]['medium']),
                  (direction['forward'], steering['right'])),

        # Regel 2: Aktiv nach freien Flächen suchen - vorwärts in Richtung des freiesten Weges
        ctrl.Rule(sector_inputs[0]['far'] & sector_inputs[5]['far'] &
                  (sector_inputs[6]['far'] | sector_inputs[7]['far']) &
                  (sector_inputs[2]['medium'] | sector_inputs[3]['medium']),
                  (direction['forward'], steering['left'])),

        # Regel 3: Vorwärts und rechts lenken, wenn linke Seite eingeschränkt
        ctrl.Rule(sector_inputs[0]['far'] &
                  (sector_inputs[6]['near'] | sector_inputs[7]['near']) &
                  (sector_inputs[2]['far'] | sector_inputs[3]['far']),
                  (direction['forward'], steering['right'])),

        # Regel 4: Vorwärts und links lenken, wenn rechte Seite eingeschränkt
        ctrl.Rule(sector_inputs[0]['far'] &
                  (sector_inputs[2]['near'] | sector_inputs[3]['near']) &
                  (sector_inputs[6]['far'] | sector_inputs[7]['far']),
                  (direction['forward'], steering['left'])),

        # Regel 5: Vorwärts rechts bei vorderer rechter Gefahr
        ctrl.Rule(sector_inputs[1]['near'] & sector_inputs[0]['far'],
                  (direction['forward'], steering['right'])),

        # Regel 6: Vorwärts links bei vorderer linker Gefahr
        ctrl.Rule(sector_inputs[7]['near'] & sector_inputs[0]['far'],
                  (direction['forward'], steering['left'])),

        # DREIPUNKTWENDE - Phase 1: Rückwärtsfahren mit starker Lenkung bei frontaler Blockade
        # Regel 7a: Dreipunktwende rechts - Phase 1: Rückwärts mit starker Rechtslenkung
        ctrl.Rule(sector_inputs[0]['near'] &
                  (sector_inputs[2]['far'] | sector_inputs[3]['far']) &
                  (sector_inputs[6]['near'] | sector_inputs[7]['near']) &
                  sector_inputs[4]['far'],
                  (direction['backward'], steering['right'])),

        # Regel 7b: Dreipunktwende links - Phase 1: Rückwärts mit starker Linkslenkung
        ctrl.Rule(sector_inputs[0]['near'] &
                  (sector_inputs[6]['far'] | sector_inputs[7]['far']) &
                  (sector_inputs[2]['near'] | sector_inputs[3]['near']) &
                  sector_inputs[4]['far'],
                  (direction['backward'], steering['left'])),

        # DREIPUNKTWENDE - Phase 2: Vorwärtsfahren mit entgegengesetzter Lenkung
        # Regel 8a: Dreipunktwende rechts - Phase 2: Vorwärts mit Linkslenkung
        ctrl.Rule(sector_inputs[0]['medium'] &
                  sector_inputs[1]['far'] &
                  (sector_inputs[2]['far'] | sector_inputs[3]['far']) &
                  sector_inputs[4]['medium'],
                  (direction['forward'], steering['left'])),

        # Regel 8b: Dreipunktwende links - Phase 2: Vorwärts mit Rechtslenkung
        ctrl.Rule(sector_inputs[0]['medium'] &
                  sector_inputs[7]['far'] &
                  (sector_inputs[6]['far'] | sector_inputs[7]['far']) &
                  sector_inputs[4]['medium'],
                  (direction['forward'], steering['right'])),

        # DREIPUNKTWENDE - Phase 3: Finale Ausrichtung mit Rückwärtsfahrt
        # Regel 9a: Dreipunktwende rechts - Phase 3: Rückwärts mit sanfter Rechtslenkung
        ctrl.Rule(sector_inputs[0]['far'] &
                  sector_inputs[1]['near'] &
                  sector_inputs[4]['near'] &
                  (sector_inputs[2]['far'] | sector_inputs[3]['far']),
                  (direction['backward'], steering['right'])),

        # Regel 9b: Dreipunktwende links - Phase 3: Rückwärts mit sanfter Linkslenkung
        ctrl.Rule(sector_inputs[0]['far'] &
                  sector_inputs[7]['near'] &
                  sector_inputs[4]['near'] &
                  (sector_inputs[6]['far'] | sector_inputs[7]['far']),
                  (direction['backward'], steering['left'])),

        # Regel 10: Kontinuierliche Vorwärtsfahrt bei freiem Weg
        ctrl.Rule(sector_inputs[0]['far'] &
                  (sector_inputs[1]['far'] | sector_inputs[7]['far']) &
                  (sector_inputs[2]['far'] | sector_inputs[3]['far']) &
                  (sector_inputs[6]['far'] | sector_inputs[7]['far']),
                  (direction['forward'], steering['straight'])),

        # Regel 11: Sanfte Korrektur bei seitlichen Hindernissen - vorwärts mit minimaler Lenkung
        ctrl.Rule(sector_inputs[0]['far'] &
                  (sector_inputs[2]['medium'] | sector_inputs[6]['medium']),
                  (direction['forward'], steering['straight'])),

        # Regel 12: Nur stoppen bei kompletter Einkesselung (sehr selten)
        ctrl.Rule(sector_inputs[0]['near'] & sector_inputs[1]['near'] &
                  sector_inputs[7]['near'] & sector_inputs[4]['near'] &
                  (sector_inputs[2]['near'] | sector_inputs[3]['near']) &
                  (sector_inputs[6]['near'] | sector_inputs[7]['near']),
                  (direction['stop'], steering['straight'])),
    ]

    # Create and store fuzzy system
    fuzzy_system = ctrl.ControlSystem(rules)
    return fuzzy_system

def execute_fuzzy_action(direction_val, steering_val):
    """Execute action based on fuzzy logic outputs using global ESC and servo values."""
    if direction_val <= -0.5:
        backward_sync()
    elif direction_val >= 0.5:
        forward()
    else:
        stop()
    
    if steering_val <= -0.5:
        left()
    elif steering_val >= 0.5:
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
    """Monitor LiDAR data for obstacles in Sector 0 (front) and Sector 4 (rear)."""
    global collision_front, collision_rear, current_direction, last_front_distance, last_rear_distance
    while safety_running:
        if lidar and lidar.is_connected:
            point_cloud = lidar.get_point_cloud()
            if point_cloud:
                adjusted_angle = lambda angle: (angle + ANGLE_OFFSET) % 360
                front_points = [p for p in point_cloud if
                                (adjusted_angle(p['angle']) >= FRONT_SECTOR_MIN or
                                 adjusted_angle(p['angle']) < FRONT_SECTOR_MAX)]
                rear_points = [p for p in point_cloud if
                               REAR_SECTOR_MIN <= adjusted_angle(p['angle']) < REAR_SECTOR_MAX]
                collision_front = len([p for p in front_points if p['distance'] < COLLISION_THRESHOLD]) >= 2
                collision_rear = len([p for p in rear_points if p['distance'] < COLLISION_THRESHOLD_REAR]) >= 2
                if front_points:
                    last_front_distance = min([p['distance'] for p in front_points])
                if rear_points:
                    last_rear_distance = min([p['distance'] for p in rear_points])
                if collision_front and current_direction == 'forward':
                    emergency_stop()
                elif collision_rear and current_direction == 'backward':
                    emergency_stop()
        time.sleep(0.2)

def lidar_worker() -> None:
    """Continuously read and parse LiDAR data."""
    global lidar_running
    while lidar_running:
        if lidar and lidar.is_connected:
            lidar.read_and_parse_data()
        time.sleep(0.1)

def autonomous_loop() -> None:
    """Run autonomous navigation loop using fuzzy logic."""
    global autonomous_running, lidar, fuzzy_system, sector_inputs
    if fuzzy_system is None:
        fuzzy_system = setup_fuzzy_logic()
    fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_system)
    print("Available fuzzy inputs:", [sd.label for sd in sector_inputs])
    while autonomous_running:
        if lidar and lidar.is_connected:
            point_cloud = lidar.get_point_cloud()
            if point_cloud:
                features = get_features(point_cloud)
                try:
                    for i, dist in enumerate(features):
                        fuzzy_sim.input[f'sector_{i}'] = dist
                    fuzzy_sim.compute()
                    direction_val = fuzzy_sim.output.get('direction', 0)
                    steering_val = fuzzy_sim.output.get('steering', 0)
                    execute_fuzzy_action(direction_val, steering_val)
                except ValueError as e:
                    print(f"Fuzzy input error: {e}")
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
                    document.getElementById('angle-range-display').textContent = `Front Safety: ${status.safety_distance_front}m, Rear Safety: ${status.safety_distance_rear}m, ESC Forward: ${status.esc_forward}, ESC Backward: ${status.esc_backward}, LiDAR: ${status.lidar_running ? 'Running' : 'Stopped'}, Safety: ${status.safety_running ? 'Running' : 'Stopped'}`;
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
                            polar: {
                                radialaxis: { visible: true, range: zoomLevels[currentZoomIndex] },
                                angularaxis: { showgrid: false, gridcolor: 'transparent', showticklabels: false, direction: 'clockwise' }
                            },
                            showlegend: false,
                            margin: { t: 20, b: 20, l: 20, r: 20 }
                        });
                        return;
                    }
                    const sectorBoundaries = [337.5, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5];
                    const sectorConnectors = [];
                    for (let i = 0; i < 8; i++) {
                        const startAngle = sectorBoundaries[i];
                        const endAngle = sectorBoundaries[(i + 1) % 8];
                        const minDistance = data.sector_distances[i];
                        sectorConnectors.push({
                            type: 'scatterpolar',
                            r: [0, minDistance, minDistance, 0],
                            theta: [startAngle, startAngle, endAngle, endAngle],
                            mode: 'lines',
                            fill: 'toself',
                            fillcolor: 'rgba(0, 255, 0, 0.2)',
                            line: { color: 'rgba(255, 0, 0, 0.5)', width: 2 }
                        });
                    }
                    Plotly.react('plot', [
                        {
                            type: 'scatterpolar',
                            r: data.distances,
                            theta: data.angles,
                            mode: 'markers',
                            marker: { color: 'blue', size: 5 }
                        },
                        ...sectorConnectors
                    ], {
                        polar: {
                            radialaxis: { visible: true, range: zoomLevels[currentZoomIndex] },
                            angularaxis: { showgrid: false, gridcolor: 'transparent', showticklabels: false, direction: 'clockwise' }
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
                const escForward = document.getElementById('esc-forward').value;
                const escBackward = document.getElementById('esc-backward').value;
                try {
                    const response = await fetch('/set_settings', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            safety_distance_front: parseFloat(safetyDistanceFront),
                            safety_distance_rear: parseFloat(safetyDistanceRear),
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
                        document.getElementById('esc-forward').value = status.esc_forward;
                        document.getElementById('esc-backward').value = status.esc_backward;
                        updateLidarStatus();
                    })
                    .catch(error => {
                        document.getElementById('status').innerText = `Error loading initial settings: ${error}`;
                    });
                setInterval(updateLidarStatus, 100);
                Plotly.newPlot('plot', [], {
                    polar: {
                        radialaxis: { visible: true, range: [0, 12] },
                        angularaxis: { showgrid: false, gridcolor: 'transparent', showticklabels: false }
                    },
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
                <input type="number" id="safety-distance-front" min="0.1" max="2.0" step="0.1" value="0.15">
            </div>
            <div>
                <label>Rear Safety Distance (m):</label>
                <input type="number" id="safety-distance-rear" min="0.1" max="2.0" step="0.1" value="0.15">
            </div>
            <div>
                <label>ESC Forward (µs):</label>
                <input type="number" id="esc-forward" min="1500" max="2000" step="10" value="1605">
            </div>
            <div>
                <label>ESC Backward (µs):</label>
                <input type="number" id="esc-backward" min="1000" max="1500" step="10" value="1310">
            </div>
            <button onclick="updateSettings()">Update Settings</button>
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

async def start_autonomous_async(request: web.Request) -> web.Response:
    """Start autonomous navigation with fuzzy logic."""
    global autonomous_running
    if not autonomous_running:
        autonomous_running = True
        threading.Thread(target=autonomous_loop, daemon=True).start()
        return web.Response(text="Autonomous started")
    return web.Response(text="Autonomous already running")

async def stop_autonomous_async(request: web.Request) -> web.Response:
    """Stop autonomous navigation."""
    global autonomous_running
    autonomous_running = False
    stop()
    return web.Response(text="Autonomous stopped")

async def set_settings(request: web.Request) -> web.Response:
    """Update configuration settings from the web interface."""
    global SAFETY_DISTANCE_FRONT, SAFETY_DISTANCE_REAR, COLLISION_THRESHOLD, COLLISION_THRESHOLD_REAR
    global ESC_FORWARD, ESC_BACKWARD
    try:
        data = await request.json()
        safety_distance_front = float(data.get('safety_distance_front', SAFETY_DISTANCE_FRONT))
        safety_distance_rear = float(data.get('safety_distance_rear', SAFETY_DISTANCE_REAR))
        esc_forward = int(data.get('esc_forward', ESC_FORWARD))
        esc_backward = int(data.get('esc_backward', ESC_BACKWARD))
        if not (0.1 <= safety_distance_front <= 2.0) or not (0.1 <= safety_distance_rear <= 2.0):
            return web.Response(text="Safety distances must be between 0.1 and 2.0 meters.", status=400)
        if not (1500 <= esc_forward <= 2000) or not (1000 <= esc_backward <= 1500):
            return web.Response(text="ESC Forward must be between 1500 and 2000 µs, ESC Backward between 1000 and 1500 µs.", status=400)
        SAFETY_DISTANCE_FRONT = safety_distance_front
        SAFETY_DISTANCE_REAR = safety_distance_rear
        COLLISION_THRESHOLD = LIDAR_TO_FRONT + SAFETY_DISTANCE_FRONT
        COLLISION_THRESHOLD_REAR = LIDAR_TO_REAR + SAFETY_DISTANCE_REAR
        ESC_FORWARD = esc_forward
        ESC_BACKWARD = esc_backward
        return web.json_response({
            "message": "Settings updated",
            "safety_distance_front": SAFETY_DISTANCE_FRONT,
            "safety_distance_rear": SAFETY_DISTANCE_REAR,
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
        "safety_distance_front": SAFETY_DISTANCE_FRONT,
        "safety_distance_rear": SAFETY_DISTANCE_REAR,
        "esc_forward": ESC_FORWARD,
        "esc_backward": ESC_BACKWARD,
        "lidar_running": lidar_running,
        "safety_running": safety_running
    }
    if lidar and lidar.is_connected:
        point_cloud = lidar.get_point_cloud()
        if point_cloud:
            status["connected"] = True
            status.update(lidar.get_status())
    return web.json_response(status)

async def get_lidar_data(request: web.Request) -> web.Response:
    """Return the current LiDAR point cloud data and sector distances for visualization."""
    if lidar and lidar.is_connected:
        point_cloud = lidar.get_point_cloud()
        if point_cloud:
            angles = [p['angle'] for p in point_cloud]
            distances = [p['distance'] for p in point_cloud]
            adjusted_angles = [(angle + ANGLE_OFFSET) % 360 for angle in angles]
            sectors = [0] * len(adjusted_angles)
            sector_boundaries = [337.5, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5]
            for i in range(8):
                start_angle = sector_boundaries[i]
                end_angle = sector_boundaries[(i + 1) % 8]
                if start_angle > end_angle:
                    mask = [angle >= start_angle or angle < end_angle for angle in adjusted_angles]
                else:
                    mask = [start_angle <= angle < end_angle for angle in adjusted_angles]
                for j, in_sector in enumerate(mask):
                    if in_sector:
                        sectors[j] = i
            sector_distances = [2.0] * 8
            for i in range(8):
                sector_points = [dist for sec, dist in zip(sectors, distances) if sec == i]
                if sector_points:
                    sector_distances[i] = min(sector_points)
            return web.json_response({
                'angles': angles,
                'distances': distances,
                'sector_distances': sector_distances
            })
        return web.json_response({'angles': [], 'distances': [], 'sector_distances': [2.0] * 8})
    return web.json_response({'angles': [], 'distances': [], 'sector_distances': [2.0] * 8})

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
    web.get('/start_autonomous', start_autonomous_async),
    web.get('/stop_autonomous', stop_autonomous_async),
    web.get('/toggle_safety', toggle_safety_async)
])

if __name__ == '__main__':
    try:
        setup_gpio()
        setup_lidar()
        setup_fuzzy_logic()
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