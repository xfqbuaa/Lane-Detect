export PYTHONPATH=~/Downloads/hubq/caffe-segnet/python/:$PYTHONPATH


python main.py \
		/home/xufq/Downloads/hubq/CargoDetectLane/dataUsed/TSD-Lane/${1} \
		/home/xufq/Downloads/hubq/CargoDetectLane/dataUsed/TSD-Lane-Info/${1}-Info.xml \
		../SegNet-Tutorial/Example_Models/segnet_model_driving_webdemo.prototxt \
		../SegNet-Tutorial/Example_Models/segnet_weights_driving_webdemo.caffemodel 
