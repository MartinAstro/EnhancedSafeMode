"""Implementation of various python metrics."""
from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import abc

import gin
import numpy as np
import six

from tf_agents.metrics import py_metric
from tf_agents.metrics.py_metrics import StreamingMetric
from tf_agents.trajectories import trajectory as traj
from tf_agents.typing import types
from tf_agents.utils import nest_utils
from tf_agents.utils import numpy_storage


from typing import Any, Iterable, Optional, Text
import numpy as np
import time

def get_eval_metrics(eval_actor):
    eval_actor.run()
    results = {}
    for metric in eval_actor.metrics:
        results[metric.name] = metric.result()
    return results

def log_eval_metrics(step, metrics):
    eval_results = (', ').join(
        '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
    print('step = {0}: {1}'.format(step, eval_results))

def deque_std(buffer, len, start_index, dtype: Optional[np.dtype] = None):
    if len == buffer.shape[0]:
      return np.std(buffer, dtype=dtype)

    assert start_index == 0
    return np.std(buffer[:len], dtype=dtype)



class TimeMetric(py_metric.PyMetric):
  """Metric to track an wall clock time.

  To increase the timer, you can __call__ it (e.g. metric_obj()).
  """

  def __init__(self, name: Text = 'Timer'):
    super(TimeMetric, self).__init__(name)
    self._np_state = numpy_storage.NumpyState()
    self.reset()

  def reset(self):
    self._np_state.start = np.float64(time.time())

  def call(self):
    self._np_state.time = np.float64(time.time()) - self._np_state.start

  def result(self) -> np.float64:
    return self._np_state.time


class AverageReturnSEMetric(StreamingMetric):
  """Computes the average undiscounted reward standard error."""

  def __init__(self,
               name: Text = 'AverageReturnSE',
               buffer_size: types.Int = 10,
               batch_size: Optional[types.Int] = None):
    """Creates an AverageReturnSEMetric."""
    self._np_state = numpy_storage.NumpyState()
    # Set a dummy value on self._np_state.episode_return so it gets included in
    # the first checkpoint (before metric is first called).
    self._np_state.episode_return = np.float64(0)
    super(AverageReturnSEMetric, self).__init__(name, buffer_size=buffer_size,
                                              batch_size=batch_size)

  def _reset(self, batch_size):
    """Resets stat gathering variables."""
    self._np_state.episode_return = np.zeros(
        shape=(batch_size,), dtype=np.float64)

  def result(self) -> np.float32:
    """Returns the value of this metric."""
    if self._buffer:
      sample_std = deque_std(self._buffer._buffer, 
                            self._buffer._len, 
                            self._buffer._start_index, 
                            dtype=np.float32)
      return sample_std/self._buffer._len
    return np.array(0.0, dtype=np.float32)


  def _batched_call(self, trajectory):
    """Processes the trajectory to update the metric.

    Args:
      trajectory: a tf_agents.trajectory.Trajectory.
    """
    episode_return = self._np_state.episode_return

    is_first = np.where(trajectory.is_first())
    episode_return[is_first] = 0

    episode_return += trajectory.reward

    is_last = np.where(trajectory.is_last())
    self.add_to_buffer(episode_return[is_last])