#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import math
from parameters import d_max, alpha_max

def get_intersect(a1, a2, b1, b2):

    s = np.vstack([a1,a2,b1,b2])
    h = np.hstack((s, np.ones((4, 1))))
    l1 = np.cross(h[0], h[1])
    l2 = np.cross(h[2], h[3])
    x, y, z = np.cross(l1, l2)
    if z == 0:
        return (float('inf'), float('inf'))
    return (x/z, y/z)

def intersection_point(point, vector, edge_start, edge_end, W, H):
    t = np.linspace(-10, 10, 2)
    line = point + t[:, np.newaxis] * vector
    intersection = get_intersect(line[0], line[1], edge_start, edge_end)
    # Check if not infinite
    if intersection[0] != float('inf'):
      # check if the intersection is within the image
      if intersection[0] >= 0 and intersection[0] <= W and intersection[1] >= 0 and intersection[1] <= H:
        return intersection

def get_intersection_points(point, vector, W, H):

  edges = [
      np.array([0, 0]),
      np.array([W, 0]),
      np.array([W, H]),
      np.array([0, H])
  ]

  intersections = []
  for i in range(4):
      edge_start = edges[i]
      edge_end = edges[(i + 1) % 4]
      intersection = intersection_point(point, vector, edge_start, edge_end, W, H)
      if intersection is not None:
          intersections.append(intersection)
  return intersections

def get_event_velocity(event_position, current_time):
  linear_system = LinearSystemSolver(event_position[0], event_position[1], current_time)
  constants_vector = linear_system.solve()
  velocity = -constants_vector / np.linalg.norm(constants_vector)
  return velocity

def update_activities(line, event):
  """
  Formula 6
  """
  delta_t = event.get_time() - line.get_time()
  if event_assigned_to_line(line, event):
    return line.get_activity()*math.exp(-np.linalg.norm(event.get_velocity())*delta_t) + 1
  else:
    return line.get_activity()*math.exp(-np.linalg.norm(event.get_velocity())*delta_t)

def event_assigned_to_line(line, event):
  """
  Formula 7
  """
  normal_to_the_line = line.normal_vector()
  condition_1 = abs(np.dot(normal_to_the_line, event.get_position()) - line.get_rho()) < d_max
  condition_2 = abs(np.dot(normal_to_the_line, event.get_velocity())) > math.cos(alpha_max)
  if condition_1 and condition_2:
    return True
  return False

def initialize_new_line_model(event):
  """
  Formula 8
  """
  #n_k = v_k
  rho_k = np.dot(event.get_position(), event.get_velocity())
  # get angle theta from the velocity vector:
  theta_k = math.atan2(event.get_velocity()[1], event.get_velocity()[0])
  new_line = Line(rho_k, theta_k, event)
  return new_line

def update_rho(line, activity):
  """
  Formula 12
  """
  rho = (line.weighted_x*math.cos(line.theta) + line.weighted_y*math.sin(line.theta))/activity
  return rho

def update_theta(line, activity):
  """
  Formula 19
  """
  a = activity*(line.weighted_y2 - line.weighted_x2) + line.weighted_x**2 - line.weighted_y**2
  b = 2*(activity*line.weighted_xy - line.weighted_x*line.weighted_y)
  beta = math.sqrt(a**2 / (b**2 + a**2))
  # choosing the solution closer to cos(line.theta)
  
  if abs(math.sqrt((1 + beta) / 2) - math.cos(line.theta)) < abs(math.sqrt((1 - beta) / 2) - math.cos(line.theta)):
    cos_theta = math.sqrt((1 + beta) / 2)
    sin_theta = math.sqrt((1 - beta) / 2)
  else:
    cos_theta = math.sqrt((1 - beta) / 2)
    sin_theta = math.sqrt((1 + beta) / 2)

  if (-b / a) > 0:
    if cos_theta < math.cos(math.pi/4):
      sin_theta = -abs(sin_theta)
    else:
      sin_theta = abs(sin_theta)
  else:
    if cos_theta < math.cos(math.pi/4):
      sin_theta = abs(sin_theta)
    else:
      sin_theta = -abs(sin_theta)
  theta = math.atan2(sin_theta, cos_theta)

  return theta

class OrientedEvent:
  def __init__(self, u, v, t, p):
    self.u = u
    self.v = v
    self.t = t
    self.p = p
  def get_position(self):
    return self.u
  def get_velocity(self):
    return self.v
  def get_time(self):
    return self.t
  def get_pixel_value(self):
    return self.p

class Line:
  def __init__(self, rho, theta,event):
    self.rho = rho
    self.theta = theta
    self.time = event.get_time()
    self.activity = 1
    self.events = [event]
    self.weighted_x = event.get_position()[0]
    self.weighted_y = event.get_position()[1]
    self.weighted_xy = event.get_position()[0]*event.get_position()[1]
    self.weighted_x2 = event.get_position()[0]**2
    self.weighted_y2 = event.get_position()[1]**2
  def update_line(self, event):
    delta_t = event.get_time() - self.time
    weight = math.exp(-np.linalg.norm(event.get_velocity())*delta_t)
    self.rho = update_rho(self, self.activity + 1)
    self.theta = update_theta(self, self.activity + 1)
    self.time = event.get_time()
    self.activity = self.activity + 1
    self.events.append(event)
    self.weighted_x *= self.weighted_x*weight
    self.weighted_x += event.get_position()[0]
    self.weighted_y *= self.weighted_y*weight
    self.weighted_y += event.get_position()[1]
    self.weighted_xy *= self.weighted_xy*weight
    self.weighted_xy += event.get_position()[0]*event.get_position()[1]
    self.weighted_x2 *= self.weighted_x2*weight
    self.weighted_x2 += event.get_position()[0]**2
    self.weighted_y2 *= self.weighted_y2*weight
    self.weighted_y2 += event.get_position()[1]**2
  def get_rho(self):
    return self.rho
  def get_theta(self):
    return self.theta
  def get_time(self):
    return self.time
  def get_activity(self):
    return self.activity
  def normal_vector(self):
    return np.array([math.cos(self.theta), math.sin(self.theta)])
  def get_average_x(self):
    return np.mean([event.get_position()[0] for event in self.events])
  def get_average_y(self):
    return np.mean([event.get_position()[1] for event in self.events])
  def get_average_xy(self):
    return np.mean([event.get_position()[0]*event.get_position()[1] for event in self.events])

class LinesList:
  def __init__(self, N):
    self.N = N
    self.lines = []
  
  def add_line(self, line):
    if len(self.lines) < self.N:
      # If the list is not full, just append the new line
      self.lines.append(line)
    else:
      # If the list is full, find the line with the lowest activity and remove it
      min_activity = min(self.lines, key=lambda x: x.get_activity())
      self.lines.remove(min_activity)
      self.lines.append(line)
  
  def get_lines(self):
    return self.lines

class LinearSystemSolver:
  def __init__(self, x, y, t):
    self.x = x
    self.y = y
    self.t = t
  
  def solve(self):
    A = np.array([[np.sum(self.x**2), np.dot(self.x, self.y)],
                  [np.dot(self.x, self.y), np.sum(self.y**2)]])
    B = np.array([np.dot(self.x, self.t), np.dot(self.y, self.t)])
    det_A = np.linalg.det(A)
    if det_A == 0:
      raise ValueError('The determinant of A is 0, the system has no solution or infinite solutions')
    x1 = (1 / det_A) * np.linalg.det(np.column_stack((B, self.y)))
    x2 = (1 / det_A) * np.linalg.det(np.column_stack((self.x, B)))
    return np.array([x1, x2])

class CircularBuffer:
    def __init__(self, n, x_pixels, y_pixels):
        self.n = n
        self.x_pixels = x_pixels
        self.y_pixels = y_pixels
        self.array_buffer = [None] * n
        self.time_buffer = [None] * n
        self.insert_index = 0

    def insert(self, array, time):
        self.array_buffer[self.insert_index] = array
        self.time_buffer[self.insert_index] = time
        self.insert_index = (self.insert_index + 1) % self.n

    def get_event_average_coordinates(self, event_x, event_y):
        # Get the indices of the nonzero elements in the arrays in the buffer
        nonzero_indices = [np.nonzero(array) for array in self.array_buffer]
        # Flatten the indices and convert them to a list of tuples
        event_positions = [(x, y, time) for indices, time in zip(nonzero_indices, self.time_buffer) for x, y in zip(indices[0], indices[1])]
        # Filter the event positions to only keep those that are within the spatial neighbourhood of the event
        events_in_neighbourhood = [(x, y, t) for x, y, t in event_positions if abs(x - event_x) <= self.x_pixels and abs(y - event_y) <= self.y_pixels]
        # Calculate the average x and y positions of the events in the spatial neighbourhood
        avg_x = np.mean([x for x, _, _ in events_in_neighbourhood])
        avg_y = np.mean([y for _, y, _ in events_in_neighbourhood])
        avg_t = np.mean([t for _, _, t in events_in_neighbourhood])
        # substract the means to the events in events_in_neighbourhood
        events_in_neighbourhood_centered = [(x - avg_x, y - avg_y, t - avg_t) for x, y, t in events_in_neighbourhood]

        return events_in_neighbourhood_centered

if __name__ == '__main__':
  print("Classes of the line detection package")