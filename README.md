# Distance_Measurement
双目测距，double camera distance measurement，机器视觉，machine vision.
# Package Requirements
python 3.6.8, numpy 1.16.4, opencv-contrib-python 4.1.1.26, opencv-python 4.1.1.26.
# Equipments
double camera linked or detached, chess board paper attached on a soild plane。just print the chess board picture in the repository.
# 一页代码，直接运行
Thanks to https://github.com/LearnTechWithUs/Stereo-Vision
参考 https://blog.csdn.net/weixin_44493841/article/details/93882273
# 一些经验
- 双目视觉目前还很难有实际应用，它受光照影响很大，在同一个公式下，不同光照对距离测量影响难以忽视，因此有的厂家会加补光灯。同时它要求测量环境纹理比较丰富，若是像白墙这样的，就很难计算视差，因为找不到特异的相对应的点，到处都是相对应的点，有的厂家加红外线，具体效果如何我也没有试过。
- 其他的baseline越长，就是说两个摄像头距离越远，能测的距离也就越远。
- 现在据我所知，双目视觉还是只能应用于工厂里头，或者特定的环境，各种变量变化比较少的，像平时生活中的自由工作环境还是很难有实际应用。
