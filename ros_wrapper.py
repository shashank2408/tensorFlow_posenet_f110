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
		self.predictedOdom.header.frame_id = "odom"

		subprocess.call("rosparam load params.yaml",shell=True)
		self.image_tf = tf.placeholder(tf.float32, [1, 224, 224, 3])
		net = PoseNet({'data': self.image_tf})

		self.p3_x = net.layers['cls3_fc_pose_xyz']
		self.p3_q = net.layers['cls3_fc_pose_wpqr']

		init = tf.initialize_all_variables()
		
		#To make tensor flow run properly on the TX2 
		#https://devtalk.nvidia.com/default/topic/1029742/jetson-tx2/tensorflow-1-6-not-working-with-jetpack-3-2/
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		
		saver = tf.train.Saver()
		self.sess = tf.Session(config=config)
		self.sess.run(init)
		saver.restore(self.sess, 'PoseNet.ckpt')
		self.listener()
		self.initial = False
		self.predicted_q_init = []
		self.predicted_p_init = []

	
	def transform(self,predict_q, predicted_x):
		# Camera- x -> right y -> down z -> front
		# Vesc- x -> front y -> left z -> up
		# rotation about z axis clockwise 90 and about x axis clockwise 90
			RtZ = np.array([[0,-1,0],[1,0,0],[0,0,1]])
			RtX = np.array([[1 ,0,0],[0,0,-1],[0,1,0]])
			trX = np.dot(RtX, np.dot(RtZ,predicted_x.T))
			q_rot = quaternion_from_euler(pi/2, 0, pi/2)
			trQ = quaternion_multiply(q_rot, predict_q)
			return trQ,trX.T


	
	def imageCallback(self,image):
		cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
		X = np.zeros((1,3,224,224))
		image = cv2.resize(cv_image, (455, 256))
		image = centeredCrop(image, 224)
		X[0][0] = image[:,:,0]
		X[0][1] = image[:,:,1]
		X[0][2] = image[:,:,2]
		self.image = np.transpose(X,(0,2,3,1))
		self.callPosenet()
		
	def callPosenet(self):
		
		with tf.Session() as sess:
		# Load the data
			feed = {self.image_tf: self.image}

			predicted_x, predicted_q = self.sess.run([self.p3_x, self.p3_q], feed_dict=feed)

			predicted_q = np.squeeze(predicted_q)
			predicted_x = np.squeeze(predicted_x)
			
			#if not self.initial:
			#	self.predicted_q_init = predicted_q
			#	self.predicted_x_init = predicted_x
			#	self.initial = True
		#predicted_q = np.subtract(predicted_q,self.predicted_q_init)
		#predicted_x = np.subtract(predicted_x,self.predicted_x_init)

			predict_q, predict_x = self.transform(predicted_q,predicted_x)

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
		
