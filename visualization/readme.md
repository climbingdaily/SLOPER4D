#  Visualization

- ## **RGB visualization**

    Under `visualization`: 

1.  Install [detectron2](https://github.com/facebookresearch/detectron2.git), [pypenGL](https://github.com/mcfletch/pyopengl.git)

2.  Change the project path and dataset path in `render_sence_base.sh`.

    ```shell
    ./render_sence_base.sh "seq002_football_001"
    # in render_sence_base.sh
    PROJ_PATH="path to code"
    DATA_PATH="path to dataset"
    ...
    conda activate detectron    # change to your conda environment name
    ...
    --draw_coco17 \				# visualize COCO17 skeleton
    --draw_coco17_kps \		    # visualize COCO17 keypoints
    --draw_smpl \				# visualize SMPL
    --draw_human_pc \			# visualize human point cloud
    --draw_scene_pc \			# visualize scene point cloud
    ```

3. Use the following command to generate videos under `rgb_datas`

    ```shell
    ./render_sence_base.sh "sequence name" # seq002_football_001 for example
    ```





- ## **SMPL visualization**
   Please refer this visualization tool [SMPL-Scene Viewer](https://github.com/climbingdaily/SMPL-Scene-Viewer),
   or [aitviewer](https://github.com/climbingdaily/aitviewer)


# License
The SLOPER4D dataset is published under the Creative [Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) License. You must attribute the work in the manner specified by the authors, you may not use this work for commercial purposes and if you alter, transform, or build upon this work, you may distribute the resulting work only under the same license. Contact us if you are interested in commercial usage.

