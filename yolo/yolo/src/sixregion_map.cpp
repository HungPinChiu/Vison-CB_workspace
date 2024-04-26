#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include <vector>

int main(int argc, char** argv) {
    ros::init(argc, argv, "draw_circles");
    ros::NodeHandle nh;

    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("/robot/map", 10);

    ros::Rate rate(10);

    while (ros::ok()) {

        std::vector<geometry_msgs::Point> positions;
        geometry_msgs::Point p1, p2, p3, p4, p5, p6;
        p1.x = 0.5; p1.y = -0.3; p1.z = 0.0;
        p2.x = 0.5; p2.y = 0.3; p2.z = 0.0;
        p3.x = 0.0; p3.y = 0.5; p3.z = 0.0;
        p4.x = 0.0; p4.y = -0.5; p4.z = 0.0;
        p5.x = -0.5; p5.y = 0.3; p5.z = 0.0;
        p6.x = -0.5; p6.y = -0.3; p6.z = 0.0;
        positions.push_back(p1);
        positions.push_back(p2);
        positions.push_back(p3);
        positions.push_back(p4);
        positions.push_back(p5);
        positions.push_back(p6);

        int id = 0; 

        for (const auto& pos : positions) {
            visualization_msgs::Marker marker;
            marker.header.frame_id = "map"; 
            marker.header.stamp = ros::Time::now();
            marker.ns = "circle";
            marker.id = id++; 
            marker.type = visualization_msgs::Marker::CYLINDER;
            marker.action = visualization_msgs::Marker::ADD;
            marker.pose.position = pos; 
            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 1.0;
            marker.scale.x = 0.25;
            marker.scale.y = 0.25;
            marker.scale.z = 0.001; 
            marker.color.a = 0.5; 
            marker.color.r = 0.0; 
            marker.color.g = 1.0;
            marker.color.b = 0.0;
            marker.lifetime = ros::Duration(); 

            marker_pub.publish(marker);
        }
        
        rate.sleep();
    }

    return 0;
}
