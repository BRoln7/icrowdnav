#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Pose, PoseArray, Point
from pedsim_msgs.msg  import TrackedPersons, TrackedPerson


class TrackPedTransform():
    def __init__(self):
        
        self.trackperson_sub = rospy.Subscriber('/isaacsim/persons_states', PoseArray, self.track_callback)
        self.track_ped_pub = rospy.Publisher('/isaacsim/tracked_persons', TrackedPersons, queue_size=10)
        self.laststamp_pose = PoseArray()
        self.buffer_init = False

    def track_callback(self, _msg:PoseArray):
        track_peds = TrackedPersons()
        person_states = _msg
        i = 0
        # init last_stamp_pose_buffer
        if not self.buffer_init:
            self.buffer_init = True
            rospy.loginfo("pose_buffer init")
            for person in person_states.poses:
                self.laststamp_pose.poses.append(Pose())
        for person in person_states.poses:
            track_ped = TrackedPerson()
            track_ped.pose.pose.position.x = person.position.x
            track_ped.pose.pose.position.y = person.position.y
            laststamp_pose = self.laststamp_pose.poses[i]
            track_ped.twist.twist.linear.x = 50*(track_ped.pose.pose.position.x - laststamp_pose.position.x)
            track_ped.twist.twist.linear.y = 50*(track_ped.pose.pose.position.y - laststamp_pose.position.y)
            # rospy.loginfo("velocity: %f, %f", track_ped.twist.twist.linear.x, track_ped.twist.twist.linear.y)
            self.laststamp_pose.poses[i].position.x = person.position.x
            self.laststamp_pose.poses[i].position.y = person.position.y
            track_peds.tracks.append(track_ped)
            i += 1

        self.track_ped_pub.publish(track_peds)


if __name__ == '__main__':
    try:
        rospy.init_node('trackped_transform')
        tp = TrackPedTransform()  
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
