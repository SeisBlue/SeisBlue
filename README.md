# SeisNN ![Version](http://img.shields.io/:Version-0.5.0dev-darkgreen.svg?style=flat-square) [![License](http://img.shields.io/:License-mit-red.svg?style=flat-square)](http://badges.mit-license.org)

 ![Notebook](http://img.shields.io/:Notebook-build.20200806-orange.svg?style=flat-square) ![Docker](http://img.shields.io/:Docker-build.20210222-blue.svg?style=flat-square)

Docs build: [![CircleCI](https://circleci.com/gh/SeisNN/SeisNN/tree/master.svg?style=svg)](https://circleci.com/gh/SeisNN/SeisNN/tree/master) [![Docker Build](https://github.com/SeisNN/SeisNN/workflows/Docker%20Image/badge.svg)](https://github.com/SeisNN/SeisNN/actions?query=workflow%3A%22Docker+Image%22) 

Github Pages: https://seisnn.github.io/SeisNN/

Deep learning seismic phase picking framework with SEISAN

![workflow](workflow.png)

![example](example.png)

---

# Warning

The code is still in the development state, API will change frequently. 

Beta version will be released soon.

Please star us for upcoming updates!

---

Prerequisite:

- S-File catalog from [SEISAN](http://seisan.info/)
- SeisComP Data Structure (SDS) database. The directory and file layout of SDS is defined as:

      SDSROOT/YEAR/NET/STA/CHAN.TYPE/NET.STA.LOC.CHAN.TYPE.YEAR.DAY

Installation:

- Follow the instructions in the [Docker](docker) folder to create a Docker container.

---

Reference:
 
 [EQTansfomer](https://www.nature.com/articles/s41467-020-17591-w) | [Github](https://github.com/smousavi05/EQTransformer)

 Mousavi, S. M., Ellsworth, W. L., Zhu, W., Chuang, L. Y., & Beroza, G. C. (2020). Earthquake transformer—an attentive deep-learning model for simultaneous earthquake detection and phase picking. Nature communications, 11(1), 1-12.

 [PhaseNet](https://doi.org/10.1093/gji/ggy423) | [Github](https://github.com/wayneweiqiang/PhaseNet)

 Zhu, W., & Beroza, G. C. (2018). PhaseNet: A Deep-Neural-Network-Based Seismic Arrival Time Picking Method. arXiv preprint arXiv:1803.03211.

 [U-net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

 Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.

 [U-net ++](https://doi.org/10.1007/978-3-030-00889-5_1) | [Github](https://github.com/MrGiovanni/UNetPlusPlus)

 Zhou, Z., Siddiquee, M. M. R., Tajbakhsh, N., & Liang, J. (2018). Unet++: A nested u-net architecture for medical image segmentation. In Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support (pp. 3-11). Springer, Cham.



---

Personal Blog (Traditional Chinese only):

[Jimmy Lab wordpress](https://jimmylab.wordpress.com/)