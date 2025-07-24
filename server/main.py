import os
import asyncio
import json
import threading
import struct
import time
import uuid  # Add this import at the top

import cv2
import numpy as np
import open3d as o3d
from matplotlib import cm # pylint: disable=import-error

global client, config

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.json')

with open(config_path, 'r') as f:
    config = json.load(f)

windows = {}

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
        array = np.frombuffer(data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (self.height, self.width, 3))
        self.image_queue.append(array)

    def tick(self):
        if time.time() - self.last_data_time > 3.0:
            print(f"Window {self.id} timed out, closing...")
            self.destroy()
            return False
            
        if self.image_queue:
            img = self.image_queue.pop(0)
            cv2.imshow(self.id, img)
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
            self.vis.create_window(
                window_name=self.id,
                width=1280,
                height=720)
            
            self.vis.get_render_option().background_color = [0.05, 0.05, 0.05]
            self.vis.get_render_option().point_size = 2

    def display(self, raw_data):
        self.last_data_time = time.time()
        self.data_queue.append(raw_data)
        self.needs_update = True

    def process_data(self):
        if not self.data_queue:
            return
        
        raw_data = self.data_queue.pop()
        self.data_queue.clear()
        
        data = np.copy(np.frombuffer(raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))

        points = data[:, :-1]
        points[:, :1] = -points[:, :1]
        
        height = points[:, 2]

        if len(height) > 0:
            height_min = np.min(height)
            height_max = np.max(height)
            if height_max > height_min:
                height_normalized = (height - height_min) / (height_max - height_min)
            else:
                height_normalized = np.zeros_like(height)
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
            self.needs_update = False
        
        should_continue = self.vis.poll_events()
        self.vis.update_renderer()
        
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

async def handle_data(reader, writer):
    try:
        while True:
            batch_header_data = await reader.readexactly(8)
            if len(batch_header_data) < 8:
                print("Incomplete batch header received")
                break
                
            sensor_count, total_size = struct.unpack('II', batch_header_data)
            print(f"Batch: {sensor_count} sensors, {total_size} bytes total")

            batch_data = await reader.readexactly(total_size)
            if len(batch_data) < total_size:
                print(f"Incomplete batch data: expected {total_size}, got {len(batch_data)}")
                break

            offset = 0
            for i in range(sensor_count):
                if offset + 64 > len(batch_data):
                    print(f"Not enough data for sensor header at offset {offset}")
                    break
                    
                sensor_id_bytes = batch_data[offset:offset+16]
                sensor_id = str(uuid.UUID(bytes=sensor_id_bytes))
                
                sensor_type = batch_data[offset+16:offset+48].decode().rstrip('\x00')
                
                frame, data_length = struct.unpack('QI', batch_data[offset+48:offset+60])
                
                width, height = struct.unpack('II', batch_data[offset+60:offset+68])
                
                header_size = 68
                
                if offset + header_size + data_length > len(batch_data):
                    print(f"Not enough data for sensor data at offset {offset}, expected {data_length} bytes")
                    break

                sensor_data = batch_data[offset+header_size:offset+header_size+data_length]
                
                offset += header_size + data_length
                
                if sensor_type == 'sensor.camera.rgb':
                    print(f"Received data from {sensor_id} ({sensor_type}) - Frame: {frame}, Size: {data_length} bytes, Dimensions: {width}x{height}")
                    if sensor_id not in windows:
                        CAMERA_RGB(sensor_id, width, height)
                    windows[sensor_id].display(sensor_data)
                    
                elif sensor_type == 'sensor.lidar.ray_cast':
                    print(f"Received data from {sensor_id} ({sensor_type}) - Frame: {frame}, Size: {data_length} bytes")
                    if sensor_id not in windows:
                        LIDAR_RAY_CAST(sensor_id)
                    windows[sensor_id].display(sensor_data)
                    
                else:
                    print(f"Received data from {sensor_id} ({sensor_type}) - Frame: {frame}, Size: {data_length} bytes")

    except asyncio.IncompleteReadError:
        print("Connection closed")
    except struct.error as e:
        print(f"Struct unpacking error: {e}")
        print(f"Offset: {offset if 'offset' in locals() else 'N/A'}")
        print(f"Batch data length: {len(batch_data) if 'batch_data' in locals() else 'N/A'}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        writer.close()
        await writer.wait_closed()

async def main():
    server = await asyncio.start_server(handle_data, config['host'], config['port'])
    addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
    print(f'Serving on {addrs}')
    print('Press Enter to stop the server...')
    
    stop_event = threading.Event()
    def input_thread():
        input()
        stop_event.set()
    
    threading.Thread(target=input_thread, daemon=True).start()
    server_task = asyncio.create_task(server.serve_forever())
    
    try:
        last_update = time.time()
        while not stop_event.is_set():
            # Update visualizations at ~60 FPS
            current_time = time.time()
            if current_time - last_update > 1.0/60.0:  # ~60 FPS
                update_visualizations()
                last_update = current_time
            
            await asyncio.sleep(0.001)  # Small sleep to prevent high CPU usage
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    finally:
        # Close all windows
        for win in list(windows.values()):
            win.destroy()

        print("Shutting down server...")
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
        server.close()
        await server.wait_closed()
        print("Server stopped.")

asyncio.run(main())