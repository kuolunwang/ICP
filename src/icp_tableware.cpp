#include <iostream>
#include <unistd.h>
#include <ros/ros.h>
#include <ros/package.h>
#include <boost/foreach.hpp>
#include <Eigen/Core>
#include <vector>
#include <string>
#include "conversion.hpp"
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Pose.h>
#include <ICP/get_object_pose.h>
#include <ICP/object_id.h>
#include <ICP/pose_con.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

// tf
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

// PCL library
#include <pcl_ros/point_cloud.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/registration/icp.h>
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudXYZRGB;
typedef pcl::PointCloud<pcl::PointXYZRGBNormal> PointCloudXYZRGBNormal;
using namespace std;
using namespace cv;

class Get_object_pose
{
private:
  ros::Publisher model_publisher;
  ros::Publisher cloud_publisher;
  ros::Publisher registered_cloud_publisher;
  ros::Publisher pose_publisher;
  ros::Subscriber model_subscriber;

  ros::ServiceServer get_object_pose_srv;
  ros::ServiceServer object_id_srv;
  ros::ServiceServer get_object_pose_con_srv;
  ros::Timer timer;

  PointCloudXYZRGB::Ptr sub_cloud;
  /*Load pre-scanned Model and observed cloud*/
  PointCloudXYZRGB::Ptr model;
  PointCloudXYZRGBNormal::Ptr registered_cloud_normal;
  PointCloudXYZRGB::Ptr registered_cloud;
  geometry_msgs::Pose final_pose;

  bool trigger;
  double fit_score;

  string model_path = "/home/kl/Pick-and-Place-with-RL/catkin_ws/src/ICP/model";
  string full_path = "";
  
  void poseBroadcaster(std::string child_frame, tf::Transform transform)
  {
    static tf::TransformBroadcaster br;
    br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "camera_color_optical_frame", child_frame));
  }

  tf::Transform getTransform(std::string target, std::string source, bool result)
  {
  /*
   * Get transform from target frame to source frame
   * [in] target: target frame name
   * [in] source: source frame name
   * [out] result: if we successfully get transformation
   */
    tf::StampedTransform stf;
    static tf::TransformListener listener;
    try
    {
      listener.waitForTransform(target, source, ros::Time(0), ros::Duration(0.5));
      listener.lookupTransform(target, source, ros::Time(0), stf);
      result = true;
    }
    catch (tf::TransformException &ex)
    {
      ROS_WARN("[%s] Can't get transform from [%s] to [%s]",
               ros::this_node::getName().c_str(),
               target.c_str(),
               source.c_str());
      result = false;
    }
    return (tf::Transform(stf.getRotation(), stf.getOrigin()));
  }

  void addNormal(PointCloudXYZRGB::Ptr cloud, PointCloudXYZRGBNormal::Ptr cloud_with_normals)
  {
    /*Add normal to PointXYZRGB
		Args:
			cloud: PointCloudXYZRGB
			cloud_with_normals: PointXYZRGBNormal
	*/
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr searchTree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    searchTree->setInputCloud(cloud);
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normalEstimator;
    normalEstimator.setInputCloud(cloud);
    normalEstimator.setSearchMethod(searchTree);
    //normalEstimator.setKSearch ( 50 );
    normalEstimator.setRadiusSearch(0.01);
    normalEstimator.compute(*normals);
    pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
    vector<int> indices;
    pcl::removeNaNNormalsFromPointCloud(*cloud_with_normals, *cloud_with_normals, indices);
    return;
  }

  void point_preprocess(PointCloudXYZRGB::Ptr cloud)
  {
    /*Preprocess point before ICP
	  Args:
		cloud: PointCloudXYZRGB
	*/
    //////////////Step1. Remove Nan////////////
    vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);

    //////////////Step2. Downsample////////////
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(0.002f, 0.002f, 0.002f);
    sor.filter(*cloud);
    copyPointCloud(*cloud, *cloud);

    //////////////Step3. Denoise//////////////
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor2;
    if (cloud->points.size() > 100)
    {
      sor2.setInputCloud(cloud);
      sor2.setMeanK(50);
      sor2.setStddevMulThresh(0.5);
      sor2.filter(*cloud);
    }
    vector<int> indices2;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, indices2);

    // build the condition
    pcl::ConditionAnd<pcl::PointXYZRGB>::Ptr range_cond(new pcl::ConditionAnd<pcl::PointXYZRGB>());
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZRGB>("x", pcl::ComparisonOps::GT, -0.15)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZRGB>("x", pcl::ComparisonOps::LT, 0.26)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZRGB>("y", pcl::ComparisonOps::GT, -0.23)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZRGB>("y", pcl::ComparisonOps::LT, 0.21)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZRGB>("z", pcl::ComparisonOps::GT, 0.0)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZRGB>("z", pcl::ComparisonOps::LT, 0.785)));
    // build the filter
    pcl::ConditionalRemoval<pcl::PointXYZRGB> condrem;
    condrem.setCondition(range_cond);
    condrem.setInputCloud(cloud);
    condrem.setKeepOrganized(true);
    // apply filter
    condrem.filter(*cloud);

    return;
  }

  Eigen::Matrix4f initial_guess(PointCloudXYZRGB::Ptr cloud_src, PointCloudXYZRGB::Ptr cloud_target)
  {
    Eigen::Vector4f src_centroid, target_centroid;
    pcl::compute3DCentroid(*cloud_src, src_centroid);
    pcl::compute3DCentroid(*cloud_target, target_centroid);
    Eigen::Matrix4f tf_tran = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f tf_rot = Eigen::Matrix4f::Identity();

    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrixNormalized(*cloud_src, src_centroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
    Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();

    Eigen::Matrix3f covariance2;
    pcl::computeCovarianceMatrixNormalized(*cloud_target, target_centroid, covariance2);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver2(covariance2, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigenVectorsPCA2 = eigen_solver2.eigenvectors();

    Eigen::Matrix3f R;
    R = eigenVectorsPCA2 * eigenVectorsPCA.inverse();
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        tf_rot(i, j) = R(i, j);

    tf_tran(0, 3) = target_centroid[0] - src_centroid[0];
    tf_tran(1, 3) = target_centroid[1] - src_centroid[1];
    tf_tran(2, 3) = target_centroid[2] - src_centroid[2];
    Eigen::Matrix4f tf = tf_rot * tf_tran;
    return tf;
  }

  Eigen::Matrix4f point_2_plane_icp(PointCloudXYZRGB::Ptr cloud_src, PointCloudXYZRGB::Ptr cloud_target, PointCloudXYZRGBNormal::Ptr trans_cloud)
  {
    PointCloudXYZRGBNormal::Ptr cloud_source_normals(new PointCloudXYZRGBNormal);
    PointCloudXYZRGBNormal::Ptr cloud_target_normals(new PointCloudXYZRGBNormal);
    addNormal(cloud_src, cloud_source_normals);
    addNormal(cloud_target, cloud_target_normals);
    pcl::IterativeClosestPointWithNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>::Ptr icp(new pcl::IterativeClosestPointWithNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>());
    icp->setMaximumIterations(500);
    icp->setTransformationEpsilon(1e-6);
    icp->setEuclideanFitnessEpsilon(1e-9);
    icp->setInputSource(cloud_source_normals); // not cloud_source, but cloud_source_trans!
    icp->setInputTarget(cloud_target_normals);

    // registration
    icp->align(*trans_cloud); // use cloud with normals for ICP

    if (icp->hasConverged())
    {
      cout << "icp score: " << icp->getFitnessScore() << endl;
      fit_score = icp->getFitnessScore();
    }
    else
      cout << "Not converged." << endl;
    Eigen::Matrix4f inverse_transformation = icp->getFinalTransformation();
    return inverse_transformation;
  }

  Eigen::Matrix4f point_2_point_icp(PointCloudXYZRGB::Ptr cloud_src, PointCloudXYZRGB::Ptr cloud_target, PointCloudXYZRGB::Ptr trans_cloud)
  {
    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB>::Ptr icp(new pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB>());
    icp->setMaximumIterations(100);
    icp->setTransformationEpsilon(1e-6);
    icp->setEuclideanFitnessEpsilon(1e-7);
    icp->setInputSource(cloud_src); // not cloud_source, but cloud_source_trans!
    icp->setInputTarget(cloud_target);

    // registration
    icp->align(*trans_cloud); // use cloud with normals for ICP

    if (icp->hasConverged())
    {
      cout << "icp score: " << icp->getFitnessScore() << endl;
      fit_score = icp->getFitnessScore();
    }
    else
      cout << "Not converged." << endl;
    Eigen::Matrix4f inverse_transformation = icp->getFinalTransformation();
    return inverse_transformation;
  }

  void cloud_cb(const sensor_msgs::PointCloud2ConstPtr &input, const sensor_msgs::ImageConstPtr &img)
  {
    PointCloudXYZRGB::Ptr cloud(new PointCloudXYZRGB);
    sensor_msgs::PointCloud2Ptr tmp;
    memcpy(&tmp, &input, sizeof(input));
    cv_bridge::CvImagePtr color_img_ptr;
    Mat mask_img;

    color_img_ptr = cv_bridge::toCvCopy(img);

    color_img_ptr->image.copyTo(mask_img);

    // get width and height of 2D point cloud data
    int width = 640;
    int height = 480;
    const float Nan_value = 0.0/0.0;

    for(int i=0; i < width; i++)
    {
      for(int j=0; j < height; j++)
      {
        // Convert from u (column / width), v (row/height) to position in array
        // where X,Y,Z data starts
        int arrayPosition = j * 20480 + i * 32; // v*Cloud.row_step + u*Cloud.point_step

        // compute position in array where x,y,z data start
        int arrayPosX = arrayPosition + 0; // X has an offset of 0
        int arrayPosY = arrayPosition + 4; // Y has an offset of 4
        int arrayPosZ = arrayPosition + 8; // Z has an offset of 8

        if(mask_img.at<uchar>(i,j,0) == 0)
        {
          memcpy(&tmp->data[arrayPosX], &Nan_value, sizeof(float));
          memcpy(&tmp->data[arrayPosY], &Nan_value, sizeof(float));
          memcpy(&tmp->data[arrayPosZ], &Nan_value, sizeof(float));
        }
      }
    }

    pcl::fromROSMsg(*tmp, *cloud); //convert from PointCloud2 to pcl point type
    point_preprocess(cloud);
    *sub_cloud = *cloud;
  }

  /*
   * Service callback
   */
  bool srv_cb(ICP::get_object_pose::Request &req, ICP::get_object_pose::Response &res)
  {
    model->header.frame_id = "camera_color_optical_frame";
    sub_cloud->header.frame_id = "camera_color_optical_frame";
    fit_score = 1.0;

    Eigen::Matrix4f tf1, tf2, final_tf;

    tf2(1,0) = 1;
    tf2(0,1) = -1;

    printf("ICP\n");
    while (fit_score > 0.005 /*|| tf2(1,0) > 0 || tf2(0,1) < 0*/)
    {
      tf2 = point_2_point_icp(model, sub_cloud, registered_cloud);
      ros::spinOnce();
    }

    final_tf = tf2;

    cout << final_tf << endl;

    tf::Transform right_arm_pose, right_arm_camera_pose, final_tf_transform = eigen2tf_full(final_tf);

    right_arm_camera_pose = getTransform("right_arm/base_link", "camera_color_optical_frame", true);
    right_arm_pose = right_arm_camera_pose * final_tf_transform;

    res.object_pose.header.frame_id = "right_arm/base_link";
    res.object_pose.header.stamp = ros::Time::now();

    res.object_pose.pose = tf2Pose(right_arm_pose);
    final_pose = res.object_pose.pose;

    model_publisher.publish(model);
    cloud_publisher.publish(sub_cloud);
    registered_cloud_publisher.publish(registered_cloud);
    pose_publisher.publish(final_pose);

    poseBroadcaster("obj_pose", final_tf_transform);

    return true;
  }

  bool object_id_cb(ICP::object_id::Request &req, ICP::object_id::Response &res)
  {
    full_path = model_path + "/" + req.object + ".pcd";
    loadModels();
    return true;
  }

  bool object_pose_con(ICP::pose_con::Request &req, ICP::pose_con::Response &res)
  {
    if(req.con == true)
    {
      trigger = true;
      res.result = "open get object pose";
    }
    else
    {
      trigger = false;
      res.result = "close get object pose";
    }

    return true;
  }

  void loadModels()
  {
    printf("Load model\n");
    pcl::io::loadPCDFile<pcl::PointXYZRGB>(full_path, *model);
    printf("Finish Load pointcloud of model\n");
  }

  void get_pose_con(const ros::TimerEvent&)
  {
    if(trigger == true)
    {
      model->header.frame_id = "camera_color_optical_frame";
      sub_cloud->header.frame_id = "camera_color_optical_frame";
      fit_score = 1.0;

      Eigen::Matrix4f tf1, tf2, final_tf;

      tf2(1,0) = 1;
      tf2(0,1) = -1;

      printf("ICP\n");
      tf2 = point_2_point_icp(model, sub_cloud, registered_cloud);

      final_tf = tf2;

      cout << final_tf << endl;

      tf::Transform right_arm_pose, right_arm_camera_pose, final_tf_transform = eigen2tf_full(final_tf);

      right_arm_camera_pose = getTransform("right_arm/base_link", "camera_color_optical_frame", true);
      right_arm_pose = right_arm_camera_pose * final_tf_transform;

      final_pose = tf2Pose(right_arm_pose);

      model_publisher.publish(model);
      cloud_publisher.publish(sub_cloud);
      registered_cloud_publisher.publish(registered_cloud);
      pose_publisher.publish(final_pose);

      poseBroadcaster("obj_pose", final_tf_transform);
    }
  }

public:
  Get_object_pose(ros::NodeHandle nh)
  {
    sub_cloud.reset(new PointCloudXYZRGB);
    model.reset(new PointCloudXYZRGB);
    registered_cloud_normal.reset(new PointCloudXYZRGBNormal);
    registered_cloud.reset(new PointCloudXYZRGB);

    model_publisher = nh.advertise<sensor_msgs::PointCloud2>("/camera/model", 1);
    cloud_publisher = nh.advertise<sensor_msgs::PointCloud2>("/camera/cloud", 1);
    registered_cloud_publisher = nh.advertise<sensor_msgs::PointCloud2>("/camera/registered_cloud", 1);
    pose_publisher = nh.advertise<geometry_msgs::Pose>("/object_pose", 1);

    // Mask rcnn mask
    message_filters::Subscriber<sensor_msgs::PointCloud2> pcl_sub(nh, "/camera/depth_registered/points", 1);
    message_filters::Subscriber<sensor_msgs::Image> mask_sub(nh, "/prediction_mask", 1);
    typedef message_filters::sync_policies::ExactTime<sensor_msgs::PointCloud2, sensor_msgs::Image> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), pcl_sub, mask_sub);
    sync.registerCallback(boost::bind(&Get_object_pose::cloud_cb, this, _1, _2));

    get_object_pose_srv = nh.advertiseService("get_object_pose", &Get_object_pose::srv_cb, this);
    object_id_srv = nh.advertiseService("object_id", &Get_object_pose::object_id_cb, this);
    get_object_pose_con_srv = nh.advertiseService("get_pose_con", &Get_object_pose::object_pose_con, this);

    timer = nh.createTimer(ros::Duration(0.5), &Get_object_pose::get_pose_con, this);
  }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "Get_object_pose");
  ros::NodeHandle nh;
  Get_object_pose foo = Get_object_pose(nh);
  while (ros::ok())
    ros::spin();
  return 0;
}
