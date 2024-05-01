import asyncio
import websockets
from pyorbbecsdk import Pipeline, FrameSet, Config, OBSensorType, OBAlignMode
import cv2
import numpy as np
from utils import frame_to_bgr_image
from processing import filter, getContours, findWallLines, estimateWallDistance
import base64
import json
from gpiozero import Servo, InputDevice
from config import *
import time

ESC_KEY = 27
PRINT_INTERVAL = 1  # seconds
MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm

CENTER = 0
RIGHT = 1
LEFT = -1
BLOBRED = 2
BLOBGREEN = -2

def get_pos_percentage(perc):
  perc = min(max(perc, -1), 1) / 2 + 0.5
  return steeringMaxLeft + steeringRange*perc

steering = Servo("GPIO12", initial_value=get_pos_percentage(0))
motor = Servo("GPIO13", initial_value=0.06)
pushbutton = InputDevice("GPIO10")

direction = 0

async def image_stream(websocket: websockets.WebSocketServerProtocol, path):

    global redMax, redMin, greenMax, greenMin, direction


    running = True

    config = Config()
    config.set_align_mode(OBAlignMode.HW_MODE)
    pipeline = Pipeline()

    steering.value = get_pos_percentage(1)
    time.sleep(0.5)
    steering.value = get_pos_percentage(-1)
    time.sleep(0.5)
    steering.value = get_pos_percentage(0)
    time.sleep(0.5)

    startbutton = False
    while not startbutton:
        startbutton =  pushbutton.is_active
        time.sleep(0.1)

    startTime = time.time_ns() // 1000000
    quadrant = 0
    last_quadrant_at = time.time_ns() // 1000000 + 500

    motor.value = 0.06

    try:
        color_profile, depth_profile = get_profiles(pipeline)
        config.enable_stream(color_profile)
        config.enable_stream(depth_profile)

    except Exception as e:
        print(e)
        return

    pipeline.start(config)

    last_center_wall_at = time.time_ns() // 1000000

    try:
        while running:


            if time.time_ns() // 1000000 - startTime > 1000:
                motor.value = speed

            frames: FrameSet = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if color_frame is None or depth_frame is None:
                continue
            
            color_image = np.flip(frame_to_bgr_image(color_frame), 0)
            depth_data = np.flip(np.flip(process_depth_frame(depth_frame), 0), 1)



            [edgesImg, blurredG, blurredR, greyImg] = filter(color_image)



            contoursG = getContours(blurredG, depth_data)
            contoursR = getContours(blurredR, depth_data)

            newLines = findWallLines(edgesImg)



            # Line visualizer
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
            limg = np.zeros(color_image.shape)
            # limg = cv2.applyColorMap(depth_data, cv2.COLORMAP_JET)
            
            # centerline
            limg = cv2.line(limg, (0, centerheight + depthOffsetY), (640, centerheight + depthOffsetY), (0, 0, 255), 2)
            for i, line in enumerate(newLines):
                x1, y1, x2, y2 = line
                limg = cv2.line(limg, (x1, y1), (x2, y2), colors[1%4], 4)
            
            viz = (np.dstack((np.zeros(blurredG.shape), blurredG, blurredR)) * 255.999) .astype(np.uint8)
            for c in contoursR:
                viz = cv2.line(viz, (c[0], 0), (c[0], 479), (0, 0, 255), 1)
            for c in contoursG:
                viz = cv2.line(viz, (c[0], 0), (c[0], 479), (0, 255, 0), 1)

            viz = cv2.addWeighted(viz, 1, limg, 1, 0, dtype=0)
            
            # end viz
            
            # find relative coordinates
            # np.arange()

            classifiedobjects = []

            for c in contoursG:
                classifiedobjects += [[BLOBGREEN, c[1], c[0], c[2]]]
            for c in contoursR:
                classifiedobjects += [[BLOBRED, c[1], c[0], c[2]]]
            
            seen_left_wall = False
            seen_right_wall = False
            seen_center_wall = False

            for line in newLines:
                x1, y1, x2, y2 = line
                d1 = estimateWallDistance(x1, y1)
                d2 = estimateWallDistance(x2, y2)

                wall_dist = (d1 + d2) / 2

                if -0.05 < np.arctan2(y2-y1, x2-x1) < 0.05:
                    classifiedobjects += [[CENTER, wall_dist]]


                    last_center_wall_at = time.time_ns() // 1000000
                    seen_center_wall = True
                
                elif (x1 + x2) / 2 - 320 > 0:
                    # print("right wall")
                    seen_right_wall = True
                    classifiedobjects += [[RIGHT, wall_dist]]
                else:
                    # print("left wall")
                    seen_left_wall = True
                    classifiedobjects += [[LEFT, wall_dist]]
            
            if direction == 0:
                if seen_left_wall and not seen_right_wall:
                    direction = 1
                    print("[ROUNDDIRECTION SET CW]")
                if seen_right_wall and not seen_left_wall:
                    direction = -1
                    print("[ROUNDDIRECTION SET CC]")

                
            steeringInputs = []

            for object in classifiedobjects:
                wall_scalar = 1/(object[1]/3800)
                wi = 0 if direction <= 0 else 1

                if object[0] == CENTER:
                    if quadrant >= stopQuadrantsCount and object[1] < 1600:
                        running = False
                        print("RUN FINISHED")
                    if object[1] < 800:
                        steeringInputs += [weigths[wi][0]*wall_scalar]

                if object[0] == RIGHT:
                    if object[1] < 1200:
                        steeringInputs += [weigths[wi][1]*wall_scalar]
                if object[0] == LEFT:
                    if object[1] < 1200:
                        steeringInputs += [weigths[wi][2]*wall_scalar]
                
                if object[0] == BLOBRED and pillars:
                    if object[3] < 800:
                        steeringInputs += [weigths[wi][3]]
                if object[0] == BLOBGREEN and pillars:
                    if object[3] < 800:
                        steeringInputs += [-weigths[wi][3]]

            steer = np.sum(np.array(steeringInputs)) / 16.0

            steering.value = get_pos_percentage(steer)


            ang = steer*np.pi

            viz = draw_steering_overlay(viz, ang)
            
            if (time.time_ns() // 1000000 - last_center_wall_at) > new_center_timeout:
                print(f"{(time.time_ns() // 1000000 - last_center_wall_at)}ms not taking because timeout is at {(time.time_ns() // 1000000 - last_quadrant_at)}")

                if (time.time_ns() // 1000000 - last_quadrant_at) > new_quadrant_timeout:
                    print(f"Quadrant turend {(time.time_ns() // 1000000 - last_quadrant_at)}ms")
                    last_quadrant_at = time.time_ns() // 1000000
                    quadrant += 1
                    print(f"------------------------- {quadrant} / {stopQuadrantsCount} ------------------------------------")

            

            a_b64 = encode_image(viz)
            # a_b64 = encode_image(greyImg)
            # b_b64 = encode_image(color_image)
            # b_b64 = encode_depth(depth_data)

            # print("Sending frames to client...")
            data = {
                "a": a_b64,
                # "b": b_b64
            }
            # await websocket.send(json.dumps(data))
            
    except KeyboardInterrupt:
        pass
    finally:
        motor.value = 0.06
        steering.value = get_pos_percentage(0)
        pipeline.stop()

def get_profiles(pipeline):
    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = profile_list.get_default_video_stream_profile()
    profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = profile_list.get_default_video_stream_profile()
    return color_profile, depth_profile

def process_depth_frame(depth_frame):
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    scale = depth_frame.get_depth_scale()

    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    depth_data = depth_data.reshape((height, width))

    depth_data = depth_data.astype(np.float32) * scale
    depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
    depth_data = depth_data.astype(np.uint16)

    # depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

    return depth_data

def draw_steering_overlay(image, steering_angle):
    # Define colors
    white = (255, 255, 255)
    green = (0, 255, 0)
    red = (0, 0, 255)
    
    # Calculate angle position on the image
    center = (image.shape[1] // 2, image.shape[0])
    angle_radius = min(image.shape[0], image.shape[1]) // 4
    angle_length = 50
    angle_rad = steering_angle
    end_point = (int(center[0] + angle_radius * np.sin(angle_rad)), int(center[1] - angle_radius * np.cos(angle_rad)))
    end_point2 = (int(end_point[0] + angle_length * np.cos(angle_rad)), int(end_point[1] + angle_length * np.sin(angle_rad)))

    # Draw steering angle overlay
    cv2.circle(image, center, angle_radius, green, 2)
    cv2.arrowedLine(image, center, end_point, red, 2)

    return image



def encode_image(image):
    retval, buffer = cv2.imencode('.jpg', image)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return base64_str

def encode_depth(depth_data):
    depth_image = cv2.applyColorMap(depth_data, cv2.COLORMAP_JET)
    retval, buffer = cv2.imencode('.jpg', depth_image)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return base64_str


start_server = websockets.serve(image_stream, "0.0.0.0", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()

