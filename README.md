# ICP

This repo is a ros package for calculate iterative closest point(ICP) with c++ pcl libray. In this repo we use tableware for example, the tableware pcd file can refer [model](https://github.com/kuolunwang/ICP/tree/main/model).

## How to use ICP repo

1. Clone this repo to your workspace.
    ```
    git clone git@github.com:kuolunwang/ICP.git
    ```
2. Re-catkin_make and source devel/setup.bash.
3. Before open ICP, you must need open camera and other perception algorithm mask, to help ICP can calculate well.
4. launch ICP file to see ICP result.
    ```
    roslaunch ICP icp.launch
    ```
    
## Topic List

| Topic Name | Topic Type | Topic Description |
|:--------:|:--------:|:--------:|
| /camera/model | [PointCloud2](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/PointCloud2.html) | origin object pcd file |
| /camera/cloud | [PointCloud2](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/PointCloud2.html) | after filtered pointcloud2 |
| /camera/registered_cloud | [PointCloud2](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/PointCloud2.html) | after ICP pointcloud2 result |
| /object_pose | [Pose](http://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/Pose.html) | after ICP object pose |

## Service List

| Service Name | Service Type | Service Description |
|:--------:|:--------:|:--------:|
| /get_object_pose | [get_object_pose.srv](https://github.com/kuolunwang/ICP/blob/main/srv/get_object_pose.srv) | get object pose after ICP |
| /object | [object_id.srv](https://github.com/kuolunwang/ICP/blob/main/srv/object_id.srv) | setting which object you want to use |
| /get_pose_con | [pose_con.srv](https://github.com/kuolunwang/ICP/blob/main/srv/pose_con.srv) | the trigger signal let ICP can continuely runing |