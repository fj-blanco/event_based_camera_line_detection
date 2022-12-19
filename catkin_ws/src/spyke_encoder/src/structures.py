#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

def get_event_velocity(event_neighbourhood_x, event_neighbourhood_y, event_neighbourhood_t):
    linear_system = LinearSystemSolver(event_neighbourhood_x, event_neighbourhood_y, event_neighbourhood_t)
    constants_vector = linear_system.solve()
    if constants_vector is not None:
        if np.linalg.norm(constants_vector) > 0:
          velocity = -constants_vector / np.linalg.norm(constants_vector)
          return velocity

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
    if not det_A == 0:
      #raise ValueError('The determinant of A is 0, the system has no solution or infinite solutions')
      # solve for the constants vector
      constants_vector = np.linalg.solve(A, B)
      return constants_vector

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
      # Get the indices of the nonzero elements in the arrays in the buffer if they are not None
      nonzero_indices = [np.nonzero(array) for array in self.array_buffer if array is not None]
      # Flatten the indices and convert them to a list of tuples
      event_positions = [(x, y, t) for indices, time in zip(nonzero_indices, self.time_buffer) for x, y, t in zip(indices[0], indices[1], [time]*len(indices[0]))]
      # Use np.where to filter the event positions that are within the spatial neighbourhood of the event
      indices = np.where((np.abs(np.array([x for x, _, _ in event_positions]) - event_x) <= self.x_pixels) & (np.abs(np.array([y for _, y, _ in event_positions]) - event_y) <= self.y_pixels))
      # Filter event_positions by the indices
      events_in_neighbourhood = [event_positions[i] for i in indices[0]]
      # Calculate the average x and y positions of the events in the spatial neighbourhood
      avg_x = np.mean([x for x, _, _ in events_in_neighbourhood])
      avg_y = np.mean([y for _, y, _ in events_in_neighbourhood])
      avg_t = np.mean([t for _, _, t in events_in_neighbourhood])
      # substract the means to the events in events_in_neighbourhood
      event_neighbourhood_x = np.array([x - avg_x for x, _, _ in events_in_neighbourhood])
      event_neighbourhood_y = np.array([y - avg_y for _, y, _ in events_in_neighbourhood])
      event_neighbourhood_t = np.array([t - avg_t for _, _, t in events_in_neighbourhood])

      return event_neighbourhood_x , event_neighbourhood_y, event_neighbourhood_t

if __name__ == '__main__':
  print("Classes of the spyke encoder package")