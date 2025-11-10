from emitter_class import Emitter
from emitter_assignments import assign_emitters
import numpy as np
global fs
fs=2e7
emitters=assign_emitters()
simulation_time=120  #120 second
time=0
while time<simulation_time:
    for emitter in emitters:
        emitter.update_position(time)
        


a=1