import os
import asyncio
import json
import threading
import struct
import carla
import uuid
import numpy as np

global server, config
global sensors

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.json')

with open(config_path, 'r') as f:
    config = json.load(f)

cache = []
sensors = {}

current_frame = 0
current_data = {}

def push_data():
    global cache, current_data, current_frame
    if current_data:
        current_data['frame'] = current_frame
        cache.insert(0, current_data.copy())
        current_data.clear()
    if len(cache) > 60:
        cache.pop()

class CARLASensor:
    def __init__(self, configuration, world):
        self.configuration = configuration

        self.id = uuid.uuid4()
        self.data = None

        blueprint = world.get_blueprint_library().find(configuration['type'])
        for key, value in configuration['attributes'].items():
            blueprint.set_attribute(key, str(value))

        position = carla.Location(x=configuration['transform']['position'][0], y=configuration['transform']['position'][1], z=configuration['transform']['position'][2])
        rotation = carla.Rotation(pitch=configuration['transform']['rotation'][0], yaw=configuration['transform']['rotation'][1], roll=configuration['transform']['rotation'][2])
        transform = carla.Transform(position, rotation)

        self.sensor = world.spawn_actor(blueprint, transform)
        self.sensor.listen(self.update)
        sensors[str(self.id)] = self

    def update(self, output):
        global current_frame, current_data
        
        if current_frame != output.frame:
            push_data()
            current_frame = output.frame

        if self.configuration["type"] == "sensor.camera.rgb":
            image_array = np.frombuffer(output.raw_data, dtype=np.uint8)
            image_array = image_array.reshape((output.height, output.width, 4))
            rgb_array = image_array[:, :, :3]
            self.data = rgb_array.tobytes()
            
            current_data[str(self.id)] = {
                'type': self.configuration['type'],
                'data': self.data,
                'metadata': {
                    'width': output.width if hasattr(output, 'width') else 0,
                    'height': output.height if hasattr(output, 'height') else 0
                }
            }
        else:
            self.data = output.raw_data
            
            current_data[str(self.id)] = {
                'type': self.configuration['type'],
                'data': self.data,
            }

    def destroy(self):
        sensors[str(self.id)] = None
        self.sensor.destroy()

def open_carla():
    try:
        client = carla.Client(config['carla']['host'], config['carla']['port'])
        client.set_timeout(config['carla']['timeout'])
        
        return client
    except Exception as e:
        print(f"Error connecting to CARLA: {e}")
        return None

async def handle_data(reader, writer):
    try:
        while True:
            try:
                data = await reader.readexactly(4)
                start_frame = struct.unpack('I', data)[0]

                print(f"Received request from {writer.get_extra_info('peername')} for data beginning frame {start_frame}.")
                
                batch_data = [frame for frame in cache if 'frame' in frame and frame['frame'] >= start_frame]
                batch_data.sort(key=lambda x: x['frame'])
                
                main_header = struct.pack('I', len(batch_data))
                writer.write(main_header)

                for frame_data in batch_data:
                    frame_header = struct.pack('QI', frame_data['frame'], len(frame_data) - 1)
                    writer.write(frame_header)
                    
                    for sensor_id, sensor_info in frame_data.items():
                        if sensor_id == 'frame':
                            continue
                        
                        sensor_type = sensor_info['type'].encode()[:32].ljust(32, b'\x00')
                        data_length = len(sensor_info['data'])
                        
                        has_metadata = 'metadata' in sensor_info
                        metadata_flag = 1 if has_metadata else 0
                        
                        sensor_header = struct.pack('16s32sII', 
                            uuid.UUID(sensor_id).bytes,
                            sensor_type,
                            data_length,
                            metadata_flag
                        )
                        writer.write(sensor_header)
                        
                        if has_metadata:
                            metadata = sensor_info['metadata']
                            width = metadata.get('width', 0)
                            height = metadata.get('height', 0)
                            metadata_data = struct.pack('II', width, height)
                            writer.write(metadata_data)
                        
                        writer.write(sensor_info['data'])
                
                await writer.drain()
                
            except asyncio.IncompleteReadError:
                print("Client disconnected")
                break
            except ConnectionResetError:
                print("Connection reset by client")
                break
                
    except Exception as e:
        print(f"Error in handle_data: {e}")
    finally:
        writer.close()
        await writer.wait_closed()

async def main():
    
    client = open_carla()

    if client:
        print("Connected to CARLA!")
    else:
        print("Failed to connect to CARLA.")
        return
  
    world = client.get_world()
    for sensor_cfg in config['sensors']:
        try:
            sensor = CARLASensor(sensor_cfg, world)
            print(f"Spawned sensor: {str(sensor.id)}")
        except Exception as e:
            print(f"Error spawning sensor: {e}")

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
        while not stop_event.is_set():
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    finally:
        for sensor in sensors.values():
            sensor.sensor.destroy()

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