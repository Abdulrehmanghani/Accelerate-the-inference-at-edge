# Accelerate the inference at edge
Recent developments in deep learning have observed remarkable performance in the AI industry with highly accurate computer vision models and it is observed that computational power is getting cheaper. But data-driven decisions in deep learning and cloud computing based systems have limitations in deployment at edge devices in real-world scenarios. Since we cannot bring edge devices to the data-centers, We can't deploy a server/GPU at the edge where we can deploy our applications at low cost embedded devices, so we need to bring AI to the edge on low cost embedded devices.

## Raspberry Pi configration for the edge inference

```
sudo raspi-config
```

Selecting the “Advanced Options” from the raspi-config menu to expand the Raspbian file system on your Raspberry Pi is important before installing OpenVINO and OpenCV.
you should select the first option, “A1. Expand File System”, hit Enter
```
sudo reboot
```
```
sudo apt-get update && sudo apt-get upgrade &&
sudo apt-get install build-essential cmake unzip pkg-config &&
sudo apt-get install libjpeg-dev libpng-dev libtiff-dev  &&
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev &&
sudo apt-get install libxvidcore-dev libx264-dev &&
sudo apt-get install libgtk-3-dev &&
sudo apt-get install libcanberra-gtk* &&
sudo apt-get install libatlas-base-dev gfortran &&
sudo apt-get install python3-dev
```
To install the openCV on raspberry Pi Use the following command 

```
sudo apt install python3-opencv
```
### Intel distribution of openvino toolkit
Open terminal and run the following command to download openvino in this project we are using 2021.3.394 version of theopenvino. 
```
wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2021.3/  ** write the latest version
```
```
tar -xf version-name
```
```
mv version-name openvino
```
```
nano ~/.bashrc
```

### OpenVINO envoirnment 
```
source ~/openvino/bin/setupvars.sh
```
```
source ~/.bashrc
```
### Set the rule for Neural Compute Stick 2
```
sudo usermod -a -G users "$(whoami)"
```
```
cd ~
```
```
sh openvino/install_dependencies/install_NCS_udev_rules.sh
```
Pre-optimized models are availble in the openvinio model zoo but in this project we are using object detection models from tensorflow model zoo.
## Model Optimiztation

Model optimization is done on the high end server or a GPU.
For the optimization install openvinoon the system requirments are availble in the given link
[Install openvino.](https://docs.openvinotoolkit.org/2021.3/openvino_docs_install_guides_installing_openvino_linux.html)

In the privious link there is also Configuration of all supported frameworks. To use tensorflow please install tensorFlow Model Optimizer prerequisites.

To download the object detection model goto [tensoflow object detection](https://github.com/tensorflow/models/tree/master/research/object_detection)
or this command in your terminal to download SSD_mobilenet_v2.
```
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
```
Run this command to uncompress the model.
```
tar -xf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
```
To optimize the model use this command in the terminal. 
```
python3 /opt/intel/openvino_2021.3.394/deployment_tools/model_optimizer/mo_tf.py --saved_model_dir ssd_mobilenet_v2_coco_2018_03_29/saved_model/ --tensorflow_object_detection_api_pipeline_config ssd_mobilenet_v2_coco_2018_03_29/pipeline.config --reverse_input_channels -o models/optimized_ssd_v2 --log_level=DEBUG --tensorflow_use_custom_operations_config /opt/intel/openvino_2021.3.394/deployment_tools/model_optimizer/extensions/front/tf/efficient_det_support_api_v2.0.json 
```
Now at the to run inference on the edge goto Raspberry Pi plugin your NCS2 and initialize the openvino envoirment in the terminal which we already have configured at the start and run this command.
```
python3 main.py -m models/optimized_ssd_v2/saved_model -i CAM -d MYRIAD -pt 0.6
```
