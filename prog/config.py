from numpy import pi

redMin = (0, 90, 80)
redMax = (5, 255, 255)
greenMin = (50, 20, 20)
greenMax = (80, 255, 255)

houghparams = {
  'threshold': 70,
  'minLineLength': 80,
  'maxLineGap': 120
}

centerstripheight = 80
centerheight = 400 // 2 + 30

minContourSize = 800
contourSizeConstant = 1

depthOffsetX = -20
depthOffsetY = -35

depthIntrinsics = [67.9 / 180 * pi, 45.3 / 180 * pi, ]
colorIntrinsics = [71.5 / 180 * pi, 56.7 / 180 * pi, ]