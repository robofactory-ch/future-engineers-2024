import cv2
import numpy as np
import math
from config import *

def filter(color_image):

  hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
  # red filter
  # red is at 0 and also 180, accounting for HSV wraparound
  rMask1 = cv2.inRange(hsv, redMin, redMax)
  redMinList = list(redMin)
  redMinList = [180 - redMax[0], redMinList[1], redMinList[2]]
  redMin2 = tuple(redMinList)
  redMaxList = list(redMax)
  redMaxList = [180, redMaxList[1], redMaxList[2]]
  redMax2 = tuple(redMaxList)
  rMask2 = cv2.inRange(hsv, redMin2, redMax2)
  rMask = cv2.bitwise_or(rMask1, rMask2)
  # green filter
  gMask = cv2.inRange(hsv, greenMin, greenMax)
  # blur images to remove noise
  blurredR = cv2.medianBlur(rMask, 5)
  blurredG = cv2.medianBlur(gMask, 5)
  grayImage = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
  blurredImg = cv2.GaussianBlur(grayImage, (3, 3), 0)
  # edge detection
  lower = 30
  upper = 90
  blackimg = cv2.inRange(blurredImg, 0, grayThresh)
  edgesImg = cv2.Canny(blackimg, lower, upper, 3)
  # combine images
  return [edgesImg, blurredG, blurredR, blackimg]


def findWallLines(edgesImg):
    lines = cv2.HoughLinesP(edgesImg, 1, np.pi/360,
                            threshold = houghparams['threshold'],
                            minLineLength = houghparams['minLineLength'],
                            maxLineGap = houghparams['maxLineGap'])
    if lines is not None:
        lines = list(lines)
    else:
        lines = []
    lines.sort(key=lambda x: x[0][0])

    filteredLines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if y1 == 0 or y2 == 0:
            continue
        if (abs(y1 - centerheight) > centerstripheight and abs(y2 - centerheight) > centerstripheight):
            continue
        if y1 < centerheight or y2 < centerheight:
            continue
        if abs(math.atan2(y2-y1, x2-x1)) > math.pi/3:
            continue
        filteredLines.append([x1 + depthOffsetX, y1 + depthOffsetY, x2 + depthOffsetX, y2 + depthOffsetY])
    return filteredLines

def getContours(imgIn: np.ndarray, depth_data: np.ndarray):
    edges = cv2.Canny(cv2.medianBlur(cv2.copyMakeBorder(imgIn[30:], 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0), 3), 30, 200)

    contours, hierarchy = cv2.findContours(edges, 
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    processedContours = []
    for contour in contours:
        size = cv2.contourArea(contour)
        if size > minContourSize:
            moment = cv2.moments(contour)
            if moment["m00"] != 0:
                x = int(moment["m10"] / moment["m00"])
                y = int(moment["m01"] / moment["m00"])
                if abs(y-centerheight) > 45:
                    continue

                _, _, w, h = cv2.boundingRect(contour)
                width = math.ceil(w * contourSizeConstant)

                z = estimatePillarDistance(h)
                # print(f"pillar at d={z}")

                processedContours.append([x, z, width])
    return processedContours


def estimateWallDistance(x, y):
    """
    Based on Intercept theorem, estimate distance of a point on a wall d(x, y, wall_height, focal_length, camera_intrinsics)

    d = (100 : h) * f



    SC: Focal distance                              to be measured mm
    SD: tbd distance                                =? mm
    AC: Hight on sensor / in pixels*dotspermm       =(var)mm
    BD: Wall hight                                  = 100 mm

    """

    wallheight = (y-centerheight) * 2
    if (wallheight <= 2): return np.Inf

    # print(1000 / (1/wallheight)) # calibration

    distancefactor = 47000.0
    d = (1 / wallheight) * distancefactor

    # print(d)

    return d


cx_angles = np.arange(-319.5, 320.5, 1, dtype="d")
cx_angles = np.arctan2(cx_angles, 450)
def getCameraAzimuth(x):
    # print("angle deg:", cx_angles[x]/np.pi*180)
    return float(cx_angles[x])

def estimatePillarDistance(h):
    """
    Based on Intercept theorem, estimate distance of a pillar of hight h

    d = (100 : h) * f



    SC: Focal distance                              to be measured mm
    SD: tbd distance                                =? mm
    AC: Hight on sensor / in pixels*dotspermm       =(var)mm
    BD: Wall hight                                  = 100 mm

    """
    if (h <= 2): return np.Inf

    print(1000 / (1/h)) # calibration

    distancefactor = 47000.0
    d = (1 / h) * distancefactor

    # print(d)

    return d

def intermediate_angle_radians(slope1_rad, slope2_rad):
    # Calculate the angle between two lines using their slopes in radians
    angle_radians = slope2_rad - slope1_rad
    # Adjust the angle to be between -pi/2 and pi/2
    while angle_radians <= -np.pi/2:
        angle_radians += np.pi
    while angle_radians > np.pi/2:
        angle_radians -= np.pi
    return -angle_radians