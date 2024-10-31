'''
This modules handles taking in an .obj file and generates synthetic images
'''

from isaacsim import SimulationApp

config = {
    "enable_cameras":True,
    "headless": True,
    "width": 600,
    "height":480,
    "add_ground_plane": True,
    "include_viewport": True}

app = SimulationApp(config)  # Change to True if you want headless mode


import carb
import os
import random
import omni.replicator.core as rep
from pathlib import Path
import time

# Parameters #
OUTPUT_DIR = os.path.abspath("./_output")

# Layer's Definition #

with rep.new_layer():

    camera = rep.create.camera(
            position=(2,2,2),
            look_at=(0,0,0)
        )

    sphere_light = rep.create.light(
        light_type="distant",
        temperature=6500,
        intensity=1000,

    )

    render_product = rep.create.render_product(camera, (1024, 1024))


    model = rep.create.cube(semantics=[('class', 'cube')],  position=(0, 0 , 0) )
                                                    
    with rep.trigger.on_frame(max_execs=10):
        with model:
                rep.modify.pose(
                    rotation=rep.distribution.uniform([-90]*3, [90]*3),
                )

    # Initialize and attach writer
    writer = rep.WriterRegistry.get("BasicWriter")

    writer.initialize(output_dir=OUTPUT_DIR, rgb=True)

    writer.attach([render_product])
    rep.orchestrator.run_until_complete()

app.close(wait_for_replicator = True)