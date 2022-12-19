#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np
from spyke_msg.msg import SpykeArrayAndImage, SpykeMessage
from structures import *
from parameters import *
import time

import multiprocessing as mp

bridge = CvBridge()
buffer = None
threshold = 100

def process_event(input_spykes, cb, current_time, oriented_events, x, y):
    position = np.array([x, y])
    pixel_value = input_spykes[x, y]
    event_neighbourhood_x, event_neighbourhood_y, event_neighbourhood_t = cb.get_event_average_coordinates(x, y)
    velocity = get_event_velocity(event_neighbourhood_x, event_neighbourhood_y, event_neighbourhood_t)
    if velocity is not None:
        oriented_event = OrientedEvent(position, velocity, current_time, pixel_value)
        oriented_events.append(oriented_event)

def from_spyke_to_oriented_events(input_spykes, n):
    cb = CircularBuffer(n, x_pixels_neighbourhood, y_pixels_neighbourhood)
    current_time = time.time()
    cb.insert(input_spykes, current_time)
    nonzero_indices = np.nonzero(input_spykes)
    oriented_events = []
    for x, y in zip(nonzero_indices[0], nonzero_indices[1]):
        process_event(input_spykes, cb, current_time, oriented_events, x, y)
    return oriented_events


def temporal_contrast(image, buffer, threshold):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(image_gray, buffer)
    #spykes = np.where(np.abs(diff) > threshold, np.sign(diff), 0)
    # Get the 10%  biggest values, set the rest to zero:
    spykes = np.where(np.abs(diff) > np.percentile(np.abs(diff), 99.96), np.sign(diff), 0)
    buffer = image_gray
    return spykes, buffer, diff

def image_processing_callback(incoming_msg, args):
    global buffer
    outgoing_msg, publisher = args
    outgoing_msg.spykes = []
    # Convert the image message to an OpenCV image
    cv_image = bridge.imgmsg_to_cv2(incoming_msg, desired_encoding='bgr8')
    if buffer is None:
        buffer = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    else:
        spykes, buffer, diff = temporal_contrast(cv_image, buffer, threshold)
        outgoing_msg.buffer = bridge.cv2_to_imgmsg(buffer, encoding='mono8')
        oriented_events = from_spyke_to_oriented_events(spykes, buffer_size)
        for oriented_event in oriented_events:
            spyke_msg = SpykeMessage()
            spyke_msg.pos_x = int(oriented_event.get_position()[0])
            spyke_msg.pos_y = int(oriented_event.get_position()[1])
            spyke_msg.v_x = oriented_event.get_velocity()[0]
            spyke_msg.v_y = oriented_event.get_velocity()[1]
            spyke_msg.time = oriented_event.get_time()
            spyke_msg.spyke_value= oriented_event.get_pixel_value()
            outgoing_msg.spykes.append(spyke_msg)

        publisher.publish(outgoing_msg)
        # change to black and white
        #cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # show the image
        #cv2.imshow('Processed Image', diff)
        #cv2.waitKey(1)

def sub_pub_node():
    rospy.init_node('image_subscriber')
    outgoing_msg = SpykeArrayAndImage()
    publisher = rospy.Publisher('/spykes_and_image', SpykeArrayAndImage, queue_size=10)
    rospy.Subscriber('/usb_cam/image_raw', Image, image_processing_callback, (outgoing_msg, publisher))
    rospy.spin()


if __name__ == '__main__':
    sub_pub_node()