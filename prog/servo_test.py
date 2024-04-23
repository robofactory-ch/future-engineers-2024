from time import sleep
from config import *
from gpiozero import Servo

def get_pos_percentage(perc):
  perc = min(max(perc, -1), 1)
  return steeringMaxLeft + steeringRange*perc

steering = Servo("GPIO12")
steering.value = get_pos_percentage(0)

percs = [0.5, 1, 0.5, 0, -0.5, -1, 0]
for p in percs:
  steering.value = get_pos_percentage(p)
  sleep(1)