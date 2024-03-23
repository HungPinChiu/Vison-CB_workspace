#include <ros/ros.h>
#include <geometry_msgs/PointStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <cmath>
#include <chrono>

class CameraPointListener {
public:
    CameraPointListener() {
        sub = nh.subscribe("/robot/objects/camera_point", 10, &CameraPointListener::pointCallback, this);
        pub = nh.advertise<visualization_msgs::MarkerArray>("/robot/objects/six_marker", 10);
        timer = nh.createTimer(ros::Duration(1.0), &CameraPointListener::timerCallback, this); // 1 second timer
    }

    void pointCallback(const geometry_msgs::PointStamped::ConstPtr& msg) {
        bool foundSimilar = false;
        int similarIndex = -1;
        double similarityThreshold = 0.1;  

        for (int i = 0; i < markerArray.markers.size(); ++i) {
            double distance = calculateDistance(markerArray.markers[i].pose.position, msg->point);
            if (distance < similarityThreshold) {
                foundSimilar = true;
                similarIndex = i;
                break;
            }
        }

        if (foundSimilar) {
            markerArray.markers[similarIndex].pose.position = msg->point;
        } else {
            visualization_msgs::Marker marker;
            marker.header = msg->header;
            marker.ns = "camera_points";
            marker.id = markerArray.markers.size();
            marker.type = visualization_msgs::Marker::SPHERE;
            marker.action = visualization_msgs::Marker::ADD;
            marker.pose.position = msg->point;
            marker.pose.orientation.w = 1.0;
            marker.scale.x = marker.scale.y = marker.scale.z = 0.1;
            marker.color.r = 1.0;
            marker.color.a = 1.0;
            markerArray.markers.push_back(marker);
        }

        pub.publish(markerArray);
    }

    double calculateDistance(const geometry_msgs::Point& p1, const geometry_msgs::Point& p2) {
        return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2) + std::pow(p1.z - p2.z, 2));
    }

    void timerCallback(const ros::TimerEvent&) {
        markerArray.markers.clear(); // Clear marker array
        pub.publish(markerArray);
    }

private:
    ros::NodeHandle nh;
    ros::Subscriber sub;
    ros::Publisher pub;
    ros::Timer timer;
    visualization_msgs::MarkerArray markerArray;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "camera_point_listener");

    CameraPointListener listener;

    ros::Rate rate(10); // 10 Hz
    while (ros::ok()) {
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
