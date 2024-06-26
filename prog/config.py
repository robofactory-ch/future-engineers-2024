from numpy import pi

pillars = True
stopQuadrantsCount = 12
headless = True

speed = -0.065
# speed = -0.10

redMin = (0, 90, 30)
redMax = (8, 255, 255)
greenMin = (20, 0, 0)
greenMax = (130, 255, 120)

houghparams = {
  'threshold': 70,
  'minLineLength': 20,
  'maxLineGap': 80
}

centerstripheight = 80
centerheight = 400 // 2 + 39

minContourSize = 800
contourSizeConstant = 1

depthOffsetX = -20
depthOffsetY = -35
depthOffsetX = 0
depthOffsetY = 0

depthIntrinsics = [67.9 / 180 * pi, 45.3 / 180 * pi, ]
colorIntrinsics = [71.5 / 180 * pi, 56.7 / 180 * pi, ]

grayThresh = 25


weigths = [[7, 3, 2, 11, -11],
          [7, 3, 2, 11, -11],
         ]

steeringMaxLeft = -0.88 # needs to be smaller
sterringMaxRight = 0.65
steeringRange = abs(steeringMaxLeft-sterringMaxRight)

new_center_timeout = 220
new_quadrant_timeout = 4500

start_override = 1600