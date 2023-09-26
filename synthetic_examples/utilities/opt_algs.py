import numpy as np
import synthetic_examples.utilities.synthesis_utilities as syn_ut


def grad_desc(xt, eta, gd, l_bd, u_bd):
    return syn_ut.box_proj(xt - eta * gd, l_bd, u_bd)


def nor_grad_desc(xt, eta, gd, l_bd, u_bd):
    if abs(gd) < np.finfo(float).eps * 10:
        return syn_ut.box_proj(xt, l_bd, u_bd)
    return syn_ut.box_proj(xt - eta * gd / abs(gd), l_bd, u_bd)