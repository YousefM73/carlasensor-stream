import os
import struct
import carla
import asyncio
import json
import uuid
import numpy as np

global client, config
global sensors

class CARLASensor:
    def __init__(self, configuration, world, writer):
        self.configuration = configuration
        self.writer = writer

        self.id = uuid.uuid4()
        self.data = None

        self.lidar_buffer = []
        self.last_rotation_timestamp = None

        rotation_freq = float(configuration.get('attributes', {}).get('rotation_frequency', '10'))
        self.rotation_period = 1.0 / rotation_freq

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
        
        if self.configuration["type"] == "sensor.camera.rgb":
            image_array = np.frombuffer(output.raw_data, dtype=np.uint8)
            image_array = image_array.reshape((output.height, output.width, 4))
            rgb_array = image_array[:, :, :3]
            self.data = rgb_array.tobytes()
        elif self.configuration["type"] == "sensor.lidar.ray_cast":
            current_timestamp = output.timestamp
            self.lidar_buffer.append(output.raw_data)
            if (self.last_rotation_timestamp is None or (current_timestamp - self.last_rotation_timestamp) >= self.rotation_period):
                
                if self.lidar_buffer:
                    combined_data = b''.join(self.lidar_buffer)
                    self.data = combined_data
                    self.lidar_buffer.clear()

                    self.lidar_buffer = []
                    self.last_rotation_timestamp = current_timestamp
            

        else:
            self.data = output.raw_data
        
        width = output.width if hasattr(output, 'width') else 0
        height = output.height if hasattr(output, 'height') else 0

        self.header = struct.pack('16s32sQIII', 
        self.id.bytes,
        self.configuration['type'].encode()[:32].ljust(32, b'\x00'),
        output.frame,
        len(self.data),
        width,
        height
        )

    def stream(self, output):

        self.writer.write(self.header + self.data)

    def destroy(self):
        sensors[str(self.id)] = None
        self.sensor.destroy()

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.json')

with open(config_path, 'r') as f:
    config = json.load(f)

def open_carla():
    try:
        client = carla.Client(config['carla']['host'], config['carla']['port'])
        client.set_timeout(config['carla']['timeout'])
        
        return client
    except Exception as e:
        print(f"Error connecting to CARLA: {e}")
        return None

sensors = {}

async def stream_data(writer):
    batch_data = b''
    sensor_count = 0
    
    for sensor in sensors.values():
        if sensor and sensor.data and sensor.header:
            batch_data += sensor.header + sensor.data
            sensor_count += 1
    
    if sensor_count > 0:
        batch_header = struct.pack('II', sensor_count, len(batch_data))
        
        writer.write(batch_header + batch_data)
        await writer.drain()

async def main():
  
    client = open_carla()

    if client:
        print("Connected to CARLA!")
    else:
        print("Failed to connect to CARLA.")
        return
  
    world = client.get_world()
    reader, writer = await asyncio.open_connection(config['host'], config['port'])

    if reader and writer:
        print("Connected to server!")
    else:
        print("Failed to connect to server.")
        return

    for sensor_cfg in config['sensors']:
        try:
            sensor = CARLASensor(sensor_cfg, world, writer)
            print(f"Spawned sensor: {str(sensor.id)}")
        except Exception as e:
            print(f"Error spawning sensor: {e}")

    task = None
    tick = config['tick']

    await asyncio.sleep(1)
    print("Sensors are running. Press CTRL+C to stop...")

    try:
        while True:
            await stream_data(writer)
            await asyncio.sleep(tick)
    except asyncio.CancelledError:
        print("Data stream cancelled.")
    except KeyboardInterrupt:
        print("Stopping data stream...")
        if task:
            task.cancel()
            await asyncio.sleep(tick)
            await task

    finally:
        if task:
            task.cancel()
            await asyncio.sleep(tick)
            await task

        for sensor in sensors.values():
            sensor.sensor.destroy()

        writer.close()
        await writer.wait_closed()
        print("Cleanup complete.")

if __name__ == '__main__':
    asyncio.run(main())