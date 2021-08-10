#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Positionwise feed forward layer definition."""

import torch


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units_1 (int): The number of hidden units of layer 1.
        hidden_units_2 (int): The number of hidden units of layer 2.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units_1, hidden_units_2, dropout_rate, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units_1)
        self.w_2 = torch.nn.Linear(hidden_units_1, hidden_units_2)
        self.w_3 = torch.nn.Linear(hidden_units_2, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        """Forward funciton."""
        return self.w_3(self.dropout(self.w_2(self.dropout(self.activation(self.w_1(x))))))
