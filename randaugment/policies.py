# coding=utf-8
# Copyright 2019 The Google UDA Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Augmentation policies found by AutoAugment."""


def get_trans_list():
  trans_list = [
      'Invert', 'Sharpness', 'AutoContrast', 'Posterize',
      'TranslateX', 'TranslateY', 'Equalize', 'Contrast', 
      'Color', 'Solarize', 'Brightness']
  return trans_list


def randaug_policies():
  trans_list = get_trans_list()
  op_list = []
  for trans in trans_list:
    for magnitude in range(1, 10):
      op_list += [(trans, 0.5, magnitude)]
  policies = []
  for op_1 in op_list:
    for op_2 in op_list:
      policies += [[op_1, op_2]]
  return policies

