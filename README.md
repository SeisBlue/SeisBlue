# SeisNN

Seismic event P phase picking project

Main framework: Obspy, Seisan, Tensorflow with Keras

Using U-net to generate pick probability

![example](example.png)

---

# Warning 

This version is unstable. Do not use now.

The code is still in the development state, API will change frequently. 

Please star us for upcoming updates!

---

Prerequisite:

- S-File catalog from [SEISAN](http://seisan.info/)
- SeisComP Data Structure (SDS) database. The directory and file layout of SDS is defined as:

      SDSROOT/YEAR/NET/STA/CHAN.TYPE/NET.STA.LOC.CHAN.TYPE.YEAR.DAY

Installation:

- Follow the instructions in the [Docker](docker) folder to create a Docker container.
- SSH into the Docker container you create.
- Clone this repo in the workspace

      git clone https://github.com/jimmy60504/SeisNN.git

In the [scripts](scripts) folder:

Preprocessing:

- Turn catalog and trace into training set
- Add coordinate 
- Scan through continuous data
  
Training:

- Pre-train
- Training
- Predict from saved model

Evaluate:

- Plot picking instances
- Calculate F1 score
- Quality control

Post-processing: (not yet)

- Output picks
- Earthquake location 
- Output s-file

Prototypes:

- Small example of some function

---

Reference:

 [PhaseNet](https://arxiv.org/abs/1803.03211)
 
 Zhu, W., & Beroza, G. C. (2018). PhaseNet: A Deep-Neural-Network-Based Seismic Arrival Time Picking Method. arXiv preprint arXiv:1803.03211.
 
 [U-net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
 
 Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.
 
 [U-net ++](https://doi.org/10.1007/978-3-030-00889-5_1)
  
 Zhou, Z., Siddiquee, M. M. R., Tajbakhsh, N., & Liang, J. (2018). Unet++: A nested u-net architecture for medical image segmentation. In Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support (pp. 3-11). Springer, Cham.
 


---

[Jimmy Lab wordpress](https://jimmylab.wordpress.com/)
 
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)
