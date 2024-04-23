import cv2
import numpy as np
import base64

# Open depth stream
colorStream = cv2.VideoCapture(cv2.CAP_V4L2)

depthStream = cv2.VideoCapture(cv2.CAP_OPENNI_DEPTH_MAP)

# Open color stream

# Check if streams are opened successfully
if not depthStream.isOpened() or not colorStream.isOpened():
    print("Error: Unable to open streams")
    exit()

# Grab a single color frame
ret, colorFrame = colorStream.read()
if not ret:
    print("Error: Unable to grab color frame")
    exit()

# Grab a single depth frame
ret, depthFrame = depthStream.read(cv2.CAP_OPENNI_DEPTH_MAP)
if not ret:
    print("Error: Unable to grab depth frame")
    exit()

# Apply the jet colormap to the depth frame
depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depthFrame, alpha=0.03), cv2.COLORMAP_JET)

# Convert frames to base64
retval, depthBuffer = cv2.imencode('.jpg', depth_colormap)
depthBase64 = base64.b64encode(depthBuffer).decode('utf-8')
print("\nDepth frame (base64 encoded):\n", depthBase64)

# Encode to base64
retval, colorBuffer = cv2.imencode('.jpg', colorFrame)
colorBase64 = base64.b64encode(colorBuffer).decode('utf-8')
print("Color frame (base64 encoded):\n", colorBase64)

# Print encoded frames to stdout

# Release the streams
depthStream.release()
colorStream.release()
