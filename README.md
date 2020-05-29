# pretrained-imagenet-pytorch

## Setup

```
conda install pytorch=1.5.0 torchvision=0.6.0 cudatoolkit=10.2 -c pytorch
git clone https://github.com/NVIDIA/apex
cd apex
git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```


## Results
 Architecture | Resolution | Accuracy@Top1 | Accuracy@Top5 |
|:-------------|-----------:|-----------------:|----------------:|
mobilenetv2_13([TensorFlow](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)) | 224 | 74.4 | 92.1 |
mobilenetv2_14([TensorFlow](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)) | 224 | 75.0 | 92.5 |
mobilenetv2_13(pytorch) | 224 |  |  |
mobilenetv2_14(pytorch) | 224 | 73.592 | 91.454 | 
inceptionv3 | 224 | 75.900 | 92.676 |
