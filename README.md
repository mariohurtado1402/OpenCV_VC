docker run -it --rm --network host -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY   -e USE_GPU=false  -v /home/mario/OpenCV_VC:/home/user/vc_ws   --privileged   ros2-slam
