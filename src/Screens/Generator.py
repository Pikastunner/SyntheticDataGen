from isaacsim import SimulationApp
import os

config = {
    "enable_cameras":True,
    "headless": True,
    "width": 600,
    "height":480,
    "add_ground_plane": False,
    "include_viewport": True,
    "open_usd": os.path.abspath("./_output/mesh_usd.usda")
    }

app = SimulationApp(config)  # Change to True if you want headless mode


import carb
import os
import random
import omni.replicator.core as rep
from pathlib import Path
import time

# Parameters #
OUTPUT_DIR = os.path.abspath("./_output")
MODEL_PATH = os.path.abspath("./_output/mesh_usd.usda")


# Layer's Definition #

with rep.new_layer():
    camera = rep.create.camera(
            position=(1,1,1),
            look_at=(0,0,0)
        )

    sphere_light = rep.create.light(
        light_type="distant",
        temperature=6500,
        intensity=1000,

    )

    render_product = rep.create.render_product(camera, (600, 600))
                       
    with rep.trigger.on_frame(max_execs=10):
        objects = rep.get.prim_at_path(path="/MyMesh")
        with objects:
            rep.modify.pose(
                position=(0, 0, 0),
                rotation=rep.distribution.uniform([-90]*3, [90]*3),
                scale=rep.distribution.uniform(0.1, 10),
            )

    # Initialize and attach writer
    writer = rep.WriterRegistry.get("BasicWriter")

    writer.initialize(output_dir=OUTPUT_DIR, rgb=True)

    writer.attach([render_product])
    rep.orchestrator.run_until_complete()


app.close(wait_for_replicator = True)