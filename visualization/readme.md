#  Visualization

## **RGB visualization**

1. Follow the [installation instruction](../src/readme.md)
3. Render all annotations on videos:
    ```shell
    root_folder=/path/to/sequence_folder
    bash ./visualization/render_scene.sh $root_folder
    ```
    Some parameters explaination in `render_sence.sh`：

    ```shell
        --draw_coco17 \				# visualize COCO17 skeleton
        --draw_coco17_kps \		    # visualize COCO17 keypoints
        --draw_smpl \				# visualize SMPL
        --draw_human_pc \			# visualize human point cloud
        --draw_scene_pc \			# visualize scene point cloud
    ```
## **SMPL visualization**
   Please refer these visualization tool [SMPL-Scene Viewer](https://github.com/climbingdaily/SMPL-Scene-Viewer),
   or [aitviewer](https://github.com/climbingdaily/aitviewer)


# License
The SLOPER4D codebase is published under the Creative [Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) License. You must attribute the work in the manner specified by the authors, you may not use this work for commercial purposes and if you alter, transform, or build upon this work, you may distribute the resulting work only under the same license. Contact us if you are interested in commercial usage.

