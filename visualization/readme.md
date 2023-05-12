#  Visualization

## **RGB visualization**

1. Follow the [installation instruction](../src/readme.md)
3. Render annotations to all frames and save them to a video:
    ```shell
    root_folder=/path/to/sequence_folder
    python ./visualization/render_scene.py $root_folder
    ```

    - Optional parameters：

    ```shell
        --index         # only render on the frame to a image
        --draw_smpl 	# render SMPL
        --draw_coco17 	# render COCO17 elements
        --draw_human_pc # render human point cloud
        --draw_scene_pc # render scene point cloud
    ```
## **SMPL visualization**
   Please refer these visualization tool [SMPL-Scene Viewer](https://github.com/climbingdaily/SMPL-Scene-Viewer),
   or [aitviewer](https://github.com/climbingdaily/aitviewer)


# License
The SLOPER4D codebase is published under the Creative [Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) License. You must attribute the work in the manner specified by the authors, you may not use this work for commercial purposes and if you alter, transform, or build upon this work, you may distribute the resulting work only under the same license. Contact us if you are interested in commercial usage.

