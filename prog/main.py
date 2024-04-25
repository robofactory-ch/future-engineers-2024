import asyncio
import websockets
from pyorbbecsdk import Pipeline, FrameSet, Config, OBSensorType, OBAlignMode
import cv2
import numpy as np
from utils import frame_to_bgr_image
from processing import filter, getContours, findWallLines, estimateWallDistance
import base64
import json
from gpiozero import Servo
from config import *

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

direction = 1

async def image_stream(websocket, path):

    global redMax, redMin, greenMax, greenMin, centerheight
    config = Config()
    config.set_align_mode(OBAlignMode.HW_MODE)
    pipeline = Pipeline()

    motor.value = -0.065

    try:
        color_profile, depth_profile = get_profiles(pipeline)
        config.enable_stream(color_profile)
        config.enable_stream(depth_profile)

    except Exception as e:
        print(e)
        return

    pipeline.start(config)

    try:
        while True:
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



            contoursG = getContours(blurredG)
            contoursR = getContours(blurredR)

            newLines, mirroredLines, lines = findWallLines(edgesImg)

            
            viz = (np.dstack((np.zeros(blurredG.shape),blurredG, blurredR)) * 255.999) .astype(np.uint8)

            for c in contoursR:
                viz = cv2.line(viz, (c[0], 0), (c[0], 479), (0, 0, 255), 1)
            for c in contoursG:
                viz = cv2.line(viz, (c[0], 0), (c[0], 479), (0, 255, 0), 1)

            # Line visualizer
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
            limg = np.zeros(color_image.shape)
            # limg = cv2.applyColorMap(depth_data, cv2.COLORMAP_JET)
            limg = cv2.cvtColor(edgesImg, cv2.COLOR_GRAY2BGR)
            
            # centerline
            # limg = cv2.line(limg, (0, centerheight + depthOffsetY), (640, centerheight + depthOffsetY), (0, 0, 255), 2)
            for i, line in enumerate(lines):
                [x1, y1, x2, y2] = line[0]
                limg = cv2.line(limg, (x1, y1), (x2, y2), colors[0%4], 4)
            # for i, line in enumerate(newLines):
            #     [x1, y1, x2, y2], [slope] = line
            #     limg = cv2.line(limg, (x1, y1), (x2, y2), colors[1%4], 4)
            # for i, line in enumerate(mirroredLines):
            #     [x1, y1, x2, y2], [slope] = line
            #     limg = cv2.line(limg, (x1, y1), (x2, y2), colors[3%4], 4)
            # end viz
            
            # find relative coordinates
            # np.arange()

            classifiedobjects = []

            # for c in contoursG:
            #     classifiedobjects += [[BLOBGREEN, c[1], c[0]]]
            # for c in contoursR:
            #     classifiedobjects += [[BLOBRED, c[1], c[0]]]

            for line in newLines:
                [x1, y1, x2, y2], [slope] = line
                d1 = estimateWallDistance(x1, y1)
                d2 = estimateWallDistance(x2, y2)

                wall_dist = (d1 + d2) / 2

                if abs((x1 + x2) / 2 - 320) < 100:
                    # print("found center wall ^^")

                    classifiedobjects += [[CENTER, wall_dist]]
                
                elif (x1 + x2) / 2 - 320 > 0:
                    # print("right wall")
                    classifiedobjects += [[RIGHT, wall_dist]]
                else:
                    # print("left wall")
                    classifiedobjects += [[LEFT, wall_dist]]
                
            steeringInputs = []

            for object in classifiedobjects:
                coeff = 1/(object[1]/3800)
                if object[0] == CENTER:
                    if object[1] < 1000:
                        steeringInputs += [-1*coeff]

                        print(f"CENTER WALL with {coeff}")
                if object[0] == RIGHT:
                    if object[1] < 900:
                        steeringInputs += [-1*coeff]
                        print(f"RIGHT WALL with {coeff}")
                if object[0] == LEFT:
                    if object[1] < 1200:
                        steeringInputs += [1.25*coeff]
                        print(f"LEFT WALL with {coeff}")
                
                if object[0] == BLOBRED:
                    if object[2] > 200:
                        pass
                        # steeringInputs += [15]
                if object[0] == BLOBGREEN:
                    if object[2] < 440:
                        pass
                        # steeringInputs += [-15]

            print(steer := np.sum(np.array(steeringInputs)) / 16.0)

            steering.value = get_pos_percentage(steer)


                # limg = cv2.putText(limg, f"{round(d1)}", (x1,y1), 0, 0.5, (255, 255, 255))
                # limg = cv2.putText(limg, f"{round(d2)}", (x2,y2), 0, 0.5, (255, 255, 255))
            
            
            # end relative coords



            

            # a_b64 = encode_image(viz)
            a_b64 = encode_image(edgesImg)
            b_b64 = encode_image(limg)
            # b_b64 = encode_depth(depth_data)

            print("Sending frames to client...")
            data = {
                "a": a_b64,
                "b": b_b64
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
    # depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
    depth_data = depth_data.astype(np.uint16)

    depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

    return depth_image

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

