#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <iostream>

using namespace std;
#define PI 3.1415926

int main(int argc, char** argv){
    ros::init(argc, argv, "tf_broadcaster");
    ros::NodeHandle node;

    if (argc < 9) {
        ROS_ERROR_STREAM("Usage: " << argv[0] << " x y z roll pitch yaw parent_frame child_frame [rate]");
        return -1;
    }

    double x, y, z, roll, pitch, yaw, rate;
    string parent_frame, child_frame;

    tf::TransformBroadcaster br;
    tf::Transform transform_Camera_to_map;
    tf::Quaternion q;

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

    // If rate then set
    if (argc > 9) {
        std::istringstream issRate(argv[9]);
        issRate >> rate;
    } else {
        rate = 3.0; // default rate
    }

    // Camera
    q.setRPY(roll*PI/180, pitch*PI/180, yaw*PI/180); // setRPY expects roll, pitch, yaw (in radians)
    transform_Camera_to_map.setOrigin(tf::Vector3(x, y, z));
    transform_Camera_to_map.setRotation(q);

    ros::Rate Rate(rate);

    while(node.ok()){
        br.sendTransform(
            tf::StampedTransform(
              transform_Camera_to_map,
              ros::Time::now(), parent_frame, child_frame));
        Rate.sleep();
    }
    return 0;
}