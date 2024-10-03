'''
This module handles the creation of the mesh object (.obj)

'''


import cv2


def generate_mesh_from_images(path: str, out:str = None):
    '''
    ## Args
    - path (str): path to the folder of images
    - out (str) (optional): output directory for mesh

    ## Return
    A .obj file that can be used within nvidia omniverse.

    ##  Example
    ```
    import generate_mesh_from_images from mesh_generate

    path = "path/to/your/image/folder"

    obj_mesh = generate_mesh_from_images(path)

    ```
    '''
    (path)

    pass

def remove_background_from_images(image):
    '''
    ## Args
    - image: image file

    ## Return
    A .obj file that can be used within nvidia omniverse.

    ##  Example
    ```
    import remove_background_from_images from mesh_generate

    image = get_image()

    image_new = remove_background_from_images(image)

    ```
    '''

    (image)

    pass
