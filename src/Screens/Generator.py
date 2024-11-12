###################################################################
#####   DO NOT CHANGE OR MOVE THESE LINES FROM THE HEADER     #####
###################################################################


# This needs to be imported first before anything else 
from isaacsim import SimulationApp
import os

config = {
    "enable_cameras":True,
    "headless": True,
    "width": 600,
    "height":480,
    "add_ground_plane": True,
    "include_viewport": True,
    "open_usd": os.path.abspath("./_output/mesh_usd.usda")
    }

app = SimulationApp(config)


###################################################################
#######################   REPLICATOR CODE    ######################
###################################################################

import omni.replicator.core as rep

import sys

# Parameters #
MODEL_PATH = os.path.abspath("./_output/mesh_usd.usda") # This doesn't change because i made it that way
OUTPUT_DIR = os.path.abspath(sys.argv[1] + "/coco_data/")

NUM_IMAGES = int(sys.argv[2])


# Perform replicator functions on a new layer
with rep.new_layer():
    ground = rep.create.plane(scale=(10, 10, 1), position=(0, 0, -0.1))
    ground.material_color = (0,0, 0)
    

    camera = rep.create.camera(
            position=(1,1,1),
            look_at=(0,0,0)
        )

    sphere_light = rep.create.light(
        light_type="distant",
        temperature=6500,
        intensity=1000,
    )

    ## Populate the scene with some shapes
    object = rep.get.prim_at_path(path="/GeneratedMesh")   # do not change
    rep.modify.semantics([("class", "generatedobj")], object)
    
    render_product = rep.create.render_product(camera, (640, 480))            
    with rep.trigger.on_frame(max_execs=NUM_IMAGES, rt_subframes=2):    # Controls the amount of executions (images) that are created
        with object:
            rep.modify.pose(
                position=(0, 0, 0),
                # The ptich pitch is constrained to be above the horizontal because it is stupid to take photos from angle where there is no point cloud data
                rotation=rep.distribution.uniform([0, -360, -360], [0, 360, 360]),
                scale=rep.distribution.uniform(8, 10),
            )

    # Initialize and attach writer
    writer = rep.WriterRegistry.get("CocoWriter")
    coco_categories = {
        "generatedobj" : {
            'name': 'generatedobj',
            'id': 0,
            'supercategory': 'geometry',
            "isthing": 1,
            "color": [
                    255,
                    0,
                    0
                ]
        }
    }


    writer.initialize(output_dir=OUTPUT_DIR, rgb=True, bounding_box_2d_tight = True, semantic_types=["class"], coco_categories=coco_categories, colorize_semantic_segmentation=True)

    writer.attach([render_product])
    rep.orchestrator.run_until_complete()



###################################################################
app.close(wait_for_replicator = True)