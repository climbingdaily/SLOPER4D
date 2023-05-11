<p align="center">

  <h1 align="center">SLOPER4D: A Scene-Aware Dataset for Global 4D Human Pose Estimation in Urban Environments (CVPR2023)</h1>
  <p align="center">
    <a href="https://climbingdaily.github.io/"><strong>Yudi Dai</strong></a>
    ·
    <a><strong>Yitai Lin </strong></a>
    ·
    <a><strong>Xiping Lin </strong></a>
    ·
    <a href="https://asc.xmu.edu.cn/t/wenchenglu"><strong>Chenglu Wen</strong></a>
    ·
    <a href="https://www.xu-lan.com/"><strong>Lan Xu</strong></a>
    ·
    <a href="https://xyyhw.top/"><strong>Hongwei Yi</strong></a>
    ·
    <a href="https://asc.xmu.edu.cn/t/shensiqi"><strong>Siqi Shen</strong></a>
    ·
    <a href="https://yuexinma.me/"><strong>Yuexin Ma</strong></a>
    ·
    <a href="http://www.cwang93.net/index_en.htm"><strong>Cheng Wang</strong></a>
  </p>

<p align="center">
  <a href="https://arxiv.org/pdf/2303.09095.pdf">
    <img src='https://img.shields.io/badge/ArXiv-green?style=for-the-badge&logo=adobeacrobatreader&logoWidth=20&logoColor=white&labelColor=c94330&color=db5a44' alt='Paper PDF'>
  </a>
  <a href="http://www.lidarhumanmotion.net/data-sloper4d/">
    <img src='https://img.shields.io/badge/Dataset-success?style=for-the-badge' alt='Dataset (Coming soon)...'>
  </a>
  <a href='http://www.lidarhumanmotion.net/sloper4d/'>
    <img src='https://img.shields.io/badge/Homepage-orange?style=for-the-badge&logo=Google%20chrome&logoColor=white&labelColor=D35400' alt='Project Page'></a>
</p>

<!-- <div align="center">
  <video width="90%" autoplay loop muted>
    <source src="./assets/teaser.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div> -->

  <div align="center">
    <img src="./assets/teaser.gif" alt="Logo" width="100%">
  </div>




## News and Updates
- *More info is coming soon…*
- **05/11/2023**: Released a [SAM-based tool](src/metaseg_demo.py) for 2D mask generation and updated the data loader example.
- **04/2023**: First part of the dataset V1.0 has released! ([Dataset](http://www.lidarhumanmotion.net/data-sloper4d/))
- **03/2023**: Initial release of the visualization Tool ([SMPL-Scene Viewer](https://github.com/climbingdaily/SMPL-Scene-Viewer)) (v1.0)

<br>
<br>
<br>

## Dataset 
- 15 sequences of 12 human subjects in 
- 10 scenes in urban environments (1k – 30k $m^2$)
- 100k+ frames multi-source data (20 Hz)
- including 2D / 3D annotations and 3D scenes;
7 km+ human motions.

*Every human subject signed permission to release their motion data for research purposes.*
<div align="center">
<img src="assets/1_data_intro_202332215833.gif" width="80%">
</div>
<div align="center">
<img src="assets/1_data_intro_2_202332215837.gif" width="80%"> 
</div>

### Dataset breakdown
| Num | Sequence     | Traj. length ($m$) | Area size ($m^2$) | Frames | Motions                                      |
| --- | ------------ | ---------------- | ---------------- | ------ | -------------------------------------------- |
| 001 | campus_001   | 908              | 13,400           | 16,202 | Jogging downhill, tying shoelaces, jumping   |
| 002 | football_002 | 221              | 200              | 4,665  | Juggling, passing, and shooting a football   |
| 003 | street_002   | 291              | 1,600            | 6,496  | Taking photos, putting on/taking off a bag   |
| 004 | library_001  | 440              | 2,300            | 9,949  | Borrowing books, reading, descending stairs |
| 005 | library_002  | 474              | 2,300            | 8,901  | Looking up, reading, returning a book        |
| 006 | library_003  | 477              | 2,300            | 8,386  | Admiring paintings, throwing rubbish, greeting |
| 007 | garden_001   | 217              | 3,000            | 5,994  | Raising hand, sitting on bench, going upstairs |
| 008 | running_001  | 392              | 8,500            | 2,000  | Running                                      |
| 009 | running_002  | 985              | 30,000           | 8,113  | Running                                      |
| 010 | park_001     | 642              | 9,300            | 12,445 | Visiting a park, walking up a small hill      |
| 011 | park_002     | 1,025            | 11,000           | 1,000  | Buying drinks, trotting, drinking             |
| 012 | square_001   | 264              | 3,200            | 6,792  | Making phone calls, waving, drinking          |
| 013 | sunlightRock001  | 386           | 1,900            | 10,116 | Climbing stairs, taking photos, walking      |
| 014 | garden_002   | 209              | 4,200            | 5,604  | Stooping, crossing a bridge, sitting cross-legged |
| 015 | plaza_001    | 365              | 2,700            | 7,989  | Admiring sculptures, eating    |   

## Data processing
Please see [processing pipeline](./src/readme.md). 

## Visualization
Please see [visualization script](./visualization/readme.md).

## More qualitative results
- Comparison between **IMU + ICP** and **our optimization** results.

<div align="center">
  <img src="./assets/2_compare_202332215842.gif" alt="Logo" width="100%">
</div>
<div align="center">
Borrowing and reading a book on a sofa.
</div>

<div align="center">
  <img src="./assets/2_compare_football_202332215855.gif" alt="Logo" width="100%">
</div>
<div align="center">
Playing football.
</div>

- Comparison between *original extrinsic parameters* and *our optimization results*.
<div align="center">
  <img src="./assets/2_compare_extrinsic_202332215849.gif" alt="Logo" width="100%">
</div>


### Cross-Dataset Evaluation
- LiDAR-based human pose estimation (HPE)

<div align="center">
  <img src="./assets/3_compare_lidarcap_20233221594.gif" alt="Logo" width="100%">
</div>

- Camera-based HPE
<div align="center">
  <img src="./assets/3_compare_vibe_20233221598.gif" alt="Logo" width="100%">
</div>


- Global Human Pose Estimation Comparison
<div align="center">
  <img src="./assets/4_compare_ghpe_glamr_202332215913.gif" alt="Logo" width="80%">
</div>

## License
The SLOPER4D codebase is published under the Creative [Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) License. You must attribute the work in the manner specified by the authors, you may not use this work for commercial purposes and if you alter, transform, or build upon this work, you may distribute the resulting work only under the same license. Contact us if you are interested in commercial usage.


## Citation
```
@inproceedings{dai2023sloper4d,
    title     = {SLOPER4D: A Scene-Aware Dataset for Global 4D Human Pose Estimation in Urban Environments},
    author    = {Dai, Yudi and Lin, YiTai and Lin, XiPing and Wen, Chenglu and Xu, Lan and Yi, Hongwei and Shen, Siqi and Ma, Yuexin and Wang, Cheng},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023}
}
```
