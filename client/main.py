import os
import struct
import asyncio
import json
import uuid
import time
import cv2
import numpy as np
import open3d as o3d
from matplotlib import cm

config = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r'))
windows = {}
last_frame = 0
frame_cache = []

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

class CAMERA_RGB:
    def __init__(self, id, width, height):
        self.id = id
        self.image_queue = []
        self.width = width
        self.height = height
        self.last_data_time = time.time()
        windows[id] = self

    def display(self, data):
        self.last_data_time = time.time()
        self.image_queue.append(np.frombuffer(data, dtype="uint8").reshape((self.height, self.width, 3)))

    def tick(self):
        if time.time() - self.last_data_time > 3.0:
            print(f"Window {self.id} timed out, closing...")
            self.destroy()
            return False
        if self.image_queue:
            cv2.imshow(self.id, self.image_queue.pop(0))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.destroy()
                return False
        return True

    def destroy(self):
        cv2.destroyWindow(self.id)
        windows.pop(self.id, None)

class LIDAR_RAY_CAST:
    def __init__(self, id):
        self.id = id
        self.point_list = o3d.geometry.PointCloud()
        self.vis = None
        self.geometry_added = False
        self.last_data_time = time.time()
        self.data_queue = []
        self.needs_update = False
        windows[id] = self

    def initialize_visualization(self):
        if self.vis is None:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name=self.id, width=1280, height=720)
            self.vis.get_render_option().background_color = [0.05, 0.05, 0.05]
            self.vis.get_render_option().point_size = 2

    def display(self, raw_data):
        self.last_data_time = time.time()
        self.data_queue.append(raw_data)
        self.needs_update = True

    def process_data(self):
        if not self.data_queue:
            return
        data = np.frombuffer(self.data_queue.pop(), dtype='f4').reshape((-1, 4))
        self.data_queue.clear()
        points = data[:, :-1].copy()
        points[:, :1] = -points[:, :1]
        height = points[:, 2]
        if len(height) > 0:
            height_min, height_max = np.min(height), np.max(height)
            height_normalized = (height - height_min) / (height_max - height_min) if height_max > height_min else np.zeros_like(height)
        else:
            height_normalized = np.array([])
        height_color = np.c_[np.interp(height_normalized, VID_RANGE, VIRIDIS[:, 0]),
                            np.interp(height_normalized, VID_RANGE, VIRIDIS[:, 1]),
                            np.interp(height_normalized, VID_RANGE, VIRIDIS[:, 2])]
        self.point_list.points = o3d.utility.Vector3dVector(points)
        self.point_list.colors = o3d.utility.Vector3dVector(height_color)
        if not self.geometry_added:
            self.vis.add_geometry(self.point_list)
            self.geometry_added = True
        self.needs_update = True

    def tick(self):
        if time.time() - self.last_data_time > 3.0:
            print(f"Window {self.id} timed out, closing...")
            self.destroy()
            return False
        if self.vis is None:
            self.initialize_visualization()
        self.process_data()
        if self.needs_update and self.geometry_added:
            self.vis.update_geometry(self.point_list)
            self.vis.update_renderer()
            self.needs_update = False
        should_continue = self.vis.poll_events()
        return should_continue

    def destroy(self):
        windows.pop(self.id, None)
        if self.vis is not None:
            self.vis.destroy_window()

def update_visualizations():
    windows_to_remove = []
    for window_id, win in list(windows.items()):
        try:
            if not win.tick():
                windows_to_remove.append(window_id)
        except Exception as e:
            print(f"Error updating window {window_id}: {e}")
            windows_to_remove.append(window_id)
    for window_id in windows_to_remove:
        if window_id in windows:
            windows[window_id].destroy()

def cache_frames(new_frames):
    global frame_cache
    current_time = time.time()
    for frame in new_frames:
        frame['timestamp'] = current_time
        frame_cache.append(frame)
    frame_cache = [f for f in frame_cache if f['timestamp'] > current_time - 5.0]

def get_frames_to_display():
    global frame_cache
    display_time = time.time() - 1.0
    ready_frames = [f for f in frame_cache if f['timestamp'] <= display_time]
    frame_cache = [f for f in frame_cache if f['timestamp'] > display_time]
    return ready_frames

def display_frame_data(frames):
    for frame_data in frames:
        for sensor_id, sensor_info in frame_data.items():
            if sensor_id in ['frame', 'timestamp']:
                continue
            if sensor_info['type'] == 'sensor.camera.rgb' and sensor_info['metadata']:
                if sensor_id not in windows:
                    CAMERA_RGB(sensor_id, sensor_info['metadata']['width'], sensor_info['metadata']['height'])
                windows[sensor_id].display(sensor_info['data'])
            elif sensor_info['type'] == 'sensor.lidar.ray_cast':
                if sensor_id not in windows:
                    LIDAR_RAY_CAST(sensor_id)
                windows[sensor_id].display(sensor_info['data'])

async def read_sensor_data(reader):
    try:
        num_frames = struct.unpack('I', await reader.readexactly(4))[0]
        frames = []
        for _ in range(num_frames):
            frame_number, num_sensors = struct.unpack('QI', await reader.readexactly(12))
            frame_data = {'frame': frame_number}
            for _ in range(num_sensors):
                sensor_id_bytes, sensor_type_bytes, data_length, metadata_flag = struct.unpack('16s32sII', await reader.readexactly(56))
                sensor_id = str(uuid.UUID(bytes=sensor_id_bytes))
                sensor_type = sensor_type_bytes.decode().rstrip('\x00')
                metadata = None
                if metadata_flag:
                    width, height = struct.unpack('II', await reader.readexactly(8))
                    metadata = {'width': width, 'height': height}
                frame_data[sensor_id] = {
                    'type': sensor_type,
                    'data': await reader.readexactly(data_length),
                    'metadata': metadata
                }
            frames.append(frame_data)
        return frames
    except Exception as e:
        print(f"Error reading sensor data: {e}")
        return []

async def receive_data(reader, writer):
    global last_frame
    writer.write(struct.pack('I', last_frame + 1))
    await writer.drain()
    data = await read_sensor_data(reader)
    if data:
        last_frame = data[-1]['frame']
        unique_sensors = set(sensor for frame in data for sensor in frame if sensor != 'frame')
        print(f"Received {len(data)} frames | Last frame: {last_frame} | Number of sensors: {len(unique_sensors)}")
        cache_frames(data)

async def main():
    reader, writer = await asyncio.open_connection(config['host'], config['port'])
    if not (reader and writer):
        print("Failed to connect to server.")
        return
    print("Connected to server!")
    print("Began receiving data. Press CTRL+C to stop...")
    try:
        while True:
            await receive_data(reader, writer)
            ready_frames = get_frames_to_display()
            if ready_frames:
                display_frame_data(ready_frames)
                update_visualizations()
            await asyncio.sleep(0.01)
    except (asyncio.CancelledError, KeyboardInterrupt):
        print("Stopping data stream...")
    finally:
        for win in list(windows.values()):
            win.destroy()
        writer.close()
        await writer.wait_closed()
        print("Cleanup complete.")

if __name__ == '__main__':
    asyncio.run(main())
