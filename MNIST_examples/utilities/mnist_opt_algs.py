from numpy.linalg import norm
import numpy as np


def grad_desc(param, dw, db, learning_rate):
    """
    Gradient descent algorithm
    :param param:
        param['w']: weight
        param['b']: bias
    :param dw: gradient of weight
    :param db: gradient of bias
    :param learning_rate: learning rate
    :return:
    """
    param['w'] -= learning_rate * dw  # gradient ascent
    param['b'] -= learning_rate * db
    return param['w'], param['b']


def normalized_grad_desc(param, dw, db, learning_rate):
    """
        Gradient descent algorithm
        :param param:
            param['w']: weight
            param['b']: bias
        :param dw: gradient of weight
        :param db: gradient of bias
        :param learning_rate: learning rate
        :return:
        """
    param['w'] -= learning_rate * dw/norm(dw.flatten())  # normalized gradient ascent
    param['b'] -= learning_rate * db/norm(db.flatten())
    return param


