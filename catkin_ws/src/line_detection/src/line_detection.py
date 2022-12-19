#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
from cv_bridge import CvBridge
from spyke_msg.msg import SpykeArrayAndImage
import time
import numpy as np
from structures import CircularBuffer, OrientedEvent, LinesList, update_activities, event_assigned_to_line, initialize_new_line_model, get_event_velocity, get_intersection_points
from parameters import x_pixels_neighbourhood, y_pixels_neighbourhood, A_up, buffer_size, number_of_lines

buffer_2 = None
diff = None
bridge = CvBridge()
lines_list = LinesList(number_of_lines)

def event_based_line_detection(oriented_events, lines_list):
    active_lines = []
    for event in oriented_events:
        lines_event_assigned_to = []
        for line in lines_list.get_lines():
            line.activity = update_activities(line, event)
            if event_assigned_to_line(line, event):
                lines_event_assigned_to.append(line)
        if len(lines_event_assigned_to) > 0:
            max_activity = max(lines_event_assigned_to, key=lambda x: x.get_activity())
            max_activity.update_line(event)
            if max_activity.activity > A_up:
                active_lines.append(max_activity)
        else:
            new_line = initialize_new_line_model(event)
            lines_list.add_line(new_line)
        return lines_list, active_lines

def from_spyke_to_oriented_events(input_spykes, n):
    cb = CircularBuffer(n, x_pixels_neighbourhood, y_pixels_neighbourhood)
    current_time = time.time()
    cb.insert(input_spykes, current_time)
    nonzero_indices = np.nonzero(input_spykes)
    oriented_events = []
    for x, y in zip(nonzero_indices[0], nonzero_indices[1]):
        position = np.array([x, y])
        pixel_value = input_spykes[x, y]
        velocity = get_event_velocity(position, current_time)
        oriented_event = OrientedEvent([position, velocity, current_time, pixel_value])
        oriented_events.append(oriented_event)
    return oriented_events

def callback(message, args):
    global buffer_2, diff
    lines_list = args
    input_spykes = message.spykes
    buffer = message.buffer
    oriented_events = []
    for input_spyke in input_spykes:
        oriented_event = OrientedEvent(np.array([input_spyke.pos_x, input_spyke.pos_y]), np.array([input_spyke.v_x, input_spyke.v_y]), input_spyke.time, input_spyke.spyke_value)
        oriented_events.append(oriented_event)
    cv_image = bridge.imgmsg_to_cv2(buffer, desired_encoding='bgr8')
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    if buffer_2 is not None:
        diff = cv2.absdiff(cv_image, buffer_2)
        buffer_2 = cv_image
    else:
        buffer_2 = cv_image
    H = cv_image.shape[0]
    W = cv_image.shape[1]
    line_detection = event_based_line_detection(oriented_events, lines_list)
    if line_detection is not None:
        lines_list, active_lines = line_detection
    for line in lines_list.get_lines():
        rho = line.rho
        theta = line.theta
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        initial_point = np.array([x0, y0])
        parallel_vector = np.array([-b, a])
        intersection_point = get_intersection_points(initial_point, parallel_vector, W, H)
        if len(intersection_point) == 2:
            color = (255, 255, int(255 * line.activity / A_up))
            if diff is not None:
                cv2.line(diff, (int(intersection_point[0][0]), int(intersection_point[0][1])), (int(intersection_point[1][0]), int(intersection_point[1][1])), color, 2)
    if diff is not None:
        cv2.imshow('Processed Image', diff)
        cv2.waitKey(1)


def subscriber_node():
    rospy.init_node('line_detection_node', anonymous=True)
    lines_list = LinesList(number_of_lines)
    rospy.Subscriber('/spykes_and_image', SpykeArrayAndImage, callback, (lines_list))
    rospy.spin()

if __name__ == '__main__':
    subscriber_node()