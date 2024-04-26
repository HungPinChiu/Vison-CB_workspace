#include <ros/ros.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <iostream>

using namespace std;
#define PI 3.14159265358979323846

int main(int argc, char** argv){
    ros::init(argc, argv, "static_tf2_broadcaster");
    ros::NodeHandle node;

    if (argc < 9) {
        ROS_ERROR_STREAM("Usage: " << argv[0] << " x y z roll pitch yaw parent_frame child_frame [rate]");
        return -1;
    }

    double x, y, z, roll, pitch, yaw;
    string parent_frame, child_frame;

    tf2_ros::StaticTransformBroadcaster static_broadcaster;
    geometry_msgs::TransformStamped static_transformStamped;

    std::istringstream issX(argv[1]), issY(argv[2]), issZ(argv[3]),
                        issRoll(argv[4]), issPitch(argv[5]), issYaw(argv[6]);
    issX >> x;
    issY >> y;
    issZ >> z;
    issRoll >> roll;
    issPitch >> pitch;
    issYaw >> yaw;

    // frame id 
    parent_frame = argv[7];
    child_frame = argv[8];

    // Set the transform properties
    static_transformStamped.header.stamp = ros::Time::now();
    static_transformStamped.header.frame_id = parent_frame;
    static_transformStamped.child_frame_id = child_frame;
    static_transformStamped.transform.translation.x = x;
    static_transformStamped.transform.translation.y = y;
    static_transformStamped.transform.translation.z = z;

    tf2::Quaternion q;
    q.setRPY(roll * PI / 180, pitch * PI / 180, yaw * PI / 180); // setRPY expects roll, pitch, yaw (in radians)
    static_transformStamped.transform.rotation.x = q.x();
    static_transformStamped.transform.rotation.y = q.y();
    static_transformStamped.transform.rotation.z = q.z();
    static_transformStamped.transform.rotation.w = q.w();

    // Send the transform
    static_broadcaster.sendTransform(static_transformStamped);

    ros::spin();
    return 0;
}
