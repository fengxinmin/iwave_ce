iWave reference software for IEEE 1857.11 Standard for Neural Network-Based Image Coding
==============================

This software package is the reference software for IEEE 1857.11 Standard for Neural Network-Based Image Coding. The reference software includes both encoder and decoder functionality.

Reference software is useful in aiding users of a video coding standard to establish and test conformance and interoperability, and to educate users and demonstrate the capabilities of the standard. For these purposes, this software is provided as an aid for the study and implementation of Neural Network-Based Image Coding.

The software has been developed by the IEEE 1857.11 Working Group.

Encode and decode instructions
---------------------------

**Note:** Pytorch 1.6.0 is required for running the software. This software does not suppert running on pure CPU machines now.

Open a command prompt on your system and change into the root directory of this project.

To encode a PNG file simply call:
```
python3 ./source/inference/encodeAPP.py --cfg_file cfg/encode.cfg
```

To decode a bitstream simply call:
```
python3 ./source/inference/decodeAPP.py --cfg_file cfg/decode.cfg
```

Training instructions
---------------------------

All models is stored in the model directory. You could run the code in ./source/train directory to get your own model.

More details could be found in doc directory.

Known issues in training
---------------------------
1. The training process of affine wavelet is unstable. You could run the training code again.
2. Some hyperparameters, such as training epoch in each stage(more details could be found in doc directory), need to be finely tuned, otherwise the performance of the published model(in model directory) may not be reached.
3. The online train technology is not integrated into the training code in train directory now. (Online train is adjust input image like preprocessing)

Contributing
---------------------------
This project welcomes contributions and suggestions.

Contributors
---------------------------
+ Haichuan Ma, hcma@mail.ustc.edu.cn
+ Cunhui Dong, dongcunh@mail.ustc.edu.cn
+ Huairui Wang, wanghr827@whu.edu.cn
+ Qiang Li, qli@mail.ustc.edu.cn
+ Haotian Zhang, zhanghaotian@mail.ustc.edu.cn
+ Chuanmin Jia, cmjia@pku.edu.cn
