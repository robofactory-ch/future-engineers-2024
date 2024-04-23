from time import sleep
from config import *
from gpiozero import Servo

def get_pos_percentage(perc):
  perc = min(max(perc, -1), 1) / 2 + 0.5
  return steeringMaxLeft + steeringRange*perc


steering = Servo("GPIO12", initial_value=get_pos_percentage(0))
motor = Servo("GPIO13", initial_value=0.06)


percs = [0.06, -0.03, 0.06, -0.04 ,0.06]
for p in percs:
  motor.value = p
  sleep(1.5)

percs2 = [0, 1, 0, -1, 0]
for p in percs2:
  steering.value = get_pos_percentage(p)
  sleep(1)
print("done")