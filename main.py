import cv2
import open3d as o3d
import numpy as np
import json
import base64
import time
from datetime import datetime

import paho.mqtt.client as mqttClient

Connected = False

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

brokerIP = None
brokerPort = 1883

frame = np.zeros((3, 1024, 768), np.uint8)
windows = {}

Flag = True

class CAMERA_RGB():
    def __init__(self, id, data):
        
        metadata = data['metadata']
        self.width = metadata['width']
        self.height = metadata['height']
        self.timestamp = metadata['timestamp']

        self.id = id
        self.data = None
        self.flag = False

        windows[self.id] = self

    def parse_frame(self, data, timestamp):
        self.data = np.frombuffer(data, dtype="uint8").reshape((self.height, self.width, 3))
        self.timestamp = timestamp

        self.flag = True

    def display(self):
        if self.flag:
            display_image = self.data.copy()
            
            dt = datetime.fromtimestamp(self.timestamp)
            timestamp_str = dt.strftime("%m/%d/%Y %I:%M:%S")
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            color = (255, 255, 255)
            thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(timestamp_str, font, font_scale, thickness)
            
            x, y = 10, text_height + 10
            cv2.rectangle(display_image, (x-5, y-text_height-5), (x+text_width+5, y+baseline+5), (0, 0, 0), -1)
            cv2.putText(display_image, timestamp_str, (x, y), font, font_scale, color, thickness)
            
            cv2.imshow(self.id, display_image)
            self.flag = False

    def cleanup(self):
        cv2.destroyWindow(self.id)

class LIDAR_RAY_CAST:
    def __init__(self, id, data):
        self.id = id
        self.data = None
        self.flag = False

        self.window = o3d.visualization.Visualizer()
        self.window.create_window(window_name=self.id)
        self.window.add_geometry(o3d.geometry.PointCloud())

        windows[self.id] = self

    def parse_frame(self, data):
        self.data = np.frombuffer(data, dtype="float32").reshape((-1, 4))
        self.flag = True

    def display(self):
        if self.flag:
            self.window.update_geometry(self.data)
            self.window.poll_events()
            self.window.update_renderer()

            self.flag = False
    
    def cleanup(self):
        self.window.destroy_window()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to broker")
        global Connected
        Connected = True
    else:
        print("Connection failed")

def on_message(client, userdata, message):
    print(f"Received message. (ID: {message.topic}, Payload Length: {len(message.payload)})")
    if len(message.payload)>0:
        data = json.loads(message.payload.decode('utf-8'))
        data['data'] = base64.b64decode(data['data'])
        window = windows[message.topic] if message.topic in windows else None

        if not window:
            if data['type'] == 'sensor.camera.rgb':
                CAMERA_RGB(message.topic, data)
            elif data['type'] == 'sensor.lidar.ray_cast':
                LIDAR_RAY_CAST(message.topic, data)

        window = windows[message.topic]

        if data['type'] == 'sensor.camera.rgb':
            window.parse_frame(data['data'], data['metadata']['timestamp'])
        else:
            return

if __name__ == "__main__":

    client = mqttClient.Client(mqttClient.CallbackAPIVersion.VERSION1, "CARLA_Sensors",clean_session=True)  # create new instance
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(brokerIP, brokerPort)

    client.loop_start()

    while Connected != True:
        time.sleep(0.5)

    client.subscribe("Stream/+", qos=0)

    cnt=0

    time.sleep(1)

    try:
        while True:
            for win in windows.values():
                win.display()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("exiting")
        for win in windows.values():
            win.cleanup()

        client.disconnect()
        client.loop_stop()