import asyncio
import websockets
from pyorbbecsdk import Pipeline, FrameSet, Config, OBSensorType, OBAlignMode
import cv2
import numpy as np
from utils import frame_to_bgr_image
from processing import filter, getCameraAzimuth, getContours, findWallLines, estimateWallDistance, intermediate_angle_radians
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
  perc = min(max(perc, -0.9999), 0.9999) / 2 + 0.5
  return min(max(steeringMaxLeft + steeringRange*perc, -0.9999), 0.9999)

steering = Servo("GPIO12", initial_value=get_pos_percentage(0))
motor = Servo("GPIO13", initial_value=0.06)
pushbutton = InputDevice("GPIO10")

direction = 0

async def image_stream(websocket: websockets.WebSocketServerProtocol, path):

    global redMax, redMin, greenMax, greenMin, direction
    print("starting")


    running = True

    config = Config()
    config.set_align_mode(OBAlignMode.HW_MODE)
    pipeline = Pipeline()

    try:
        color_profile, depth_profile = get_profiles(pipeline)
        config.enable_stream(color_profile)
        config.enable_stream(depth_profile)

    except Exception as e:
        print(e)
        return

    pipeline.start(config)

    print("Ready")

    steering.value = get_pos_percentage(1)
    time.sleep(0.5)
    steering.value = get_pos_percentage(-1)
    time.sleep(0.5)
    steering.value = get_pos_percentage(0)
    time.sleep(0.5)

    startbutton = True
    while not startbutton:
        startbutton =  pushbutton.is_active
        time.sleep(0.1)

    startTime = time.time_ns() // 1000000
    quadrant = 0
    last_quadrant_at = time.time_ns() // 1000000 - 250

    motor.value = 0.06

    integral_steering = 0.0

    anti_pillaring = 0.0

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

            if time.time_ns() // 1000000 - startTime < 1000:
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
            blobs = []
            if pillars:
                for c in contoursG:
                    blobs += [[BLOBGREEN, c[0], c[1], c[2]]]
                for c in contoursR:
                    blobs += [[BLOBRED, c[0], c[1], c[2]]]

            
            mellowest_left = 90.0
            mellowest_right = 90.0
            seen_center_wall = False
            classifiedobjects = []

            for line in newLines:
                x1, y1, x2, y2 = line

                # print("------")

                d1 = estimateWallDistance(x1, y1)
                d2 = estimateWallDistance(x2, y2)

                u1 = (np.tan(getCameraAzimuth(x1))) * d1
                v1 = d1

                # print(u1, v1)
                u2 = (np.tan(getCameraAzimuth(x2))) * d2
                v2 = d2

                A = (u2 - u1)
                B = (v1 - v2)
                C = (u1 * v2 - u2 * v1)

                D = np.sqrt(A**2 + B**2)

                wall_dist = -1.0
                if D != 0.0:
                    wall_dist = abs(C / D)
                
                wall_rotation = np.arctan2(B, A)
                # print(wall_rotation)
                
                # print("wall_dist", wall_dist)
                

                last_center_wall_at = time.time_ns() // 1000000
                if -0.05 < np.arctan2(y2-y1, x2-x1) < 0.05:
                    classifiedobjects += [[CENTER, wall_dist]]


                    seen_center_wall = True
                    # print("center wall at", wall_dist)
                
                elif (x1 + x2) / 2 - 320 > 0:
                    # print("right wall")
                    mellowest_right = min(abs(np.arctan2(y2-y1, x2-x1)), mellowest_right)
                    classifiedobjects += [[RIGHT, wall_dist, wall_rotation]]
                    # print("right wall at ", wall_dist)
                else:
                    # print("left wall")
                    mellowest_left = min(abs(np.arctan2(y2-y1, x2-x1)), mellowest_left)
                    classifiedobjects += [[LEFT, wall_dist, wall_rotation]]
                    # print("left wall at  ", wall_dist)
            
            if direction == 0:
                # print(mellowest_left, mellowest_right)
                if mellowest_left < mellowest_right and (mellowest_left + mellowest_right) < 179.0:
                    direction = 1
                    print("[ROUNDDIRECTION SET CW]")
                else:
                    direction = -1
                    print("[ROUNDDIRECTION SET CC]")

            # direction = 1
                
            steeringInputs = [0.00]




            pillarOverride = False
            steeringInputsP = [0.00]
            if pillars:
                for obj in blobs:
                    redgreen, x, y, width = obj
                    print("pillar", y)
                    if redgreen == BLOBRED:
                        # red
                        print(y)
                        if y < 1200:
                            steeringInputsP += [9 * float(x)/640.0]
                            pillarOverride = True
                    else:
                        # green
                        if y < 1200:
                            
                            pillarOverride = True
                            steeringInputsP += [-9 * float(640-x)/640.0]
            steeringInputs = [0.00]
            for object in classifiedobjects:
                wi = 0 if direction <= 0 else 1

                if object[0] == CENTER:
                    if quadrant >= stopQuadrantsCount and object[1] < 2300 and (integral_steering <= 0.30) and (time.time_ns() // 1000000 - last_quadrant_at) > new_quadrant_timeout / 3 and not pillars:
                        running = False
                        print("RUN FINISHED")
                    print("center", object[1])

                    if object[1] < 1175 and not pillars:
                        steeringInputs += [10 * direction]
                    if object[1] < 2000 and pillars:
                        steeringInputs += [5 * direction]
                        

                        
                pid_distance = 320 if not pillars else 370
                pid_sp_angle_r = -1.6 if not pillars else -1.6
                pid_sp_angle_l = 1.3 if not pillars else 1.3

                kp = -8.5 if not pillars and not pillarOverride else -10.5

                        
                if object[0] == RIGHT and direction == 1 and object[1] < 200:
                    steeringInputs += [weigths[wi][2] if not pillars else weigths[wi][2]*2]
                if object[0] == LEFT and direction == 1:
                    err = (pid_distance - object[1]) / 500 if quadrant > 0 else 0.0
                    ang_err = intermediate_angle_radians(pid_sp_angle_r, object[2]) / (np.pi/2)

                    steeringInputs += [err * 4.5 + ang_err * -8.5]
                

                if object[0] == LEFT and direction == -1 and object[1] < 200:
                    steeringInputs += [weigths[wi][2] if not pillars else weigths[wi][2]*2]
                if object[0] == RIGHT and direction == -1:
                    err = (pid_distance - object[1]) / 500 if quadrant > 0 else 0.0

                    ang_err = intermediate_angle_radians(pid_sp_angle_l, object[2]) / (np.pi/2)
                    print(ang_err, err)
                    steeringInputs += [err * -4.5 + ang_err * -8.5]

            steer = np.sum(np.array(steeringInputs)) / 8
            if pillarOverride:
                pval = np.sum(np.array(steeringInputsP)) / 2
                anti_pillaring += anti_pillaring * 0.75 + pval
                steer = steer*0.75 + pval
            else:
                steer = steer + (-anti_pillaring*0.75)
                # steer = steer * np.log(anti_pillaring)
                anti_pillaring *= 0.07
                print("ap:", anti_pillaring)

            if (time.time_ns() // 1000000 - startTime) < start_override and not pillars:
                steer = -direction*4
            if start_override < (time.time_ns() // 1000000 - startTime) < start_override+350 and not pillars:
                steer = direction*4

            steer = get_pos_percentage(steer)
            steer = 0.0 if np.isnan(steer) else steer

            integral_steering = integral_steering * 0.3 +  steer
            pillarOverride = False
            try:
                steering.value = steer
            except:
                print("exceptional steering: ", steer)

            if (abs(integral_steering) >= 0.90):
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
            await websocket.send(json.dumps(data))
            
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

