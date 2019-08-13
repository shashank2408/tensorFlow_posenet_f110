import rospy
import cv2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from sensor_msgs.msg import Image 
import subprocess
import numpy as np
import random
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from posenet import GoogLeNet as PoseNet
import math
import rosparam
from cv_bridge import CvBridge, CvBridgeError
from test import *

class RosWrapperPoseNet:
	def __init__(self):
		rospy.init_node("PoseNet_ROS")
		self.bridge = CvBridge()
		self.predictedOdom  = Odometry()
		self.predictedOdom.header.stamp = rospy.Time.now()
		self.predictedOdom.header.frame_id = "predictedOdom"
		#self.vescOdom = Odometry()
		#self.vescOdom.header.stamp = rospy.Time.now()
		#self.vescOdom.header.frame_id = "vescOdom"
		subprocess.call("rosparam load params.yaml",shell=True)
		self.image_tf = tf.placeholder(tf.float32, [1, 224, 224, 3])
		net = PoseNet({'data': self.image_tf})

		self.p3_x = net.layers['cls3_fc_pose_xyz']
		self.p3_q = net.layers['cls3_fc_pose_wpqr']

		init = tf.initialize_all_variables()

		saver = tf.train.Saver()
		self.sess = tf.Session()
		self.sess.run(init)
		saver.restore(self.sess, 'PoseNet.ckpt')
		self.listener()
			
	
	def imageCallback(self,image):
		cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
		mean = np.zeros((1,3,224,224))
		image = cv2.resize(cv_image, (455, 256))
		image = centeredCrop(self.image, 224)
		self.callPosenet()
		
	def callPosenet(self):
		
		with tf.Session() as sess:
		# Load the data
			feed = {self.image_tf: self.image}

			predicted_x, predicted_q = self.sess.run([self.p3_x, self.p3_q], feed_dict=feed)

			predicted_q = np.squeeze(predicted_q)
			predicted_x = np.squeeze(predicted_x)
			
		self.predictedOdom.pose.pose = Pose(Point(predicted_x[0],predicted_x[1],predicted_x[2]),\
						Quaternion(predicted_q[0],predicted_q[1],predicted_q[2],predicted_q[3]))
		self.odomPub.publish(self.predictedOdom)

	def listener(self):
		cameraTopic = rosparam.get_param("camera_topic")
		cameraSub = rospy.Subscriber(cameraTopic, Image, self.imageCallback, queue_size = 10)
		self.odomPub = rospy.Publisher("predicted_odom", Odometry, queue_size = 1)
		#odomSub = rospy.Subscriber(rosparam.get_param("odom_topic"),Odometry, self.odomCallBack)
	
if __name__=="__main__":
	ptf = RosWrapperPoseNet()
	rospy.spin()
		
