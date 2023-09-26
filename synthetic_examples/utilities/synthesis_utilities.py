import matplotlib.pyplot as plt
import numpy as np
import pickle
"""
f_t(x) = (a[t]*x-b[t])**2 / 2
b is generated in different plans through generate_opt()
"""


def generate_opt(plan, T, **kwargs):
    """
    Different plans to generate bt.
    :param plan: ['constant', 'random', 'line', 'jagged', 'cos', 'leap', 'leap_b']
    :param T: time to stop
    :param kwargs: for different plans, see kwargs.pop in each condition.
    :return:
    """
    print(plan)
    print(T)
    if plan == 'constant':
        cst = kwargs.pop('cst')
        b = np.ones(T) * cst
    elif plan == 'random':
        rand_seed = kwargs.pop('rand_seed')
        np.random.seed(rand_seed)
        u = kwargs.pop('upper_rand')
        l = kwargs.pop('lower_rand')
        assert u > l
        b = l + np.random.rand(T) * (u - l)
    elif plan == 'line':
        b = []
        tmp_T = kwargs.pop('tmp_T')
        l_bd = kwargs.pop('lower_bd')
        u_bd = kwargs.pop('upper_bd')
        repetition = int(T / tmp_T) + 1
        b_tmp = np.arange(start=l_bd, stop=u_bd, step=(u_bd - l_bd) / tmp_T)
        for i in range(repetition):
            b = np.concatenate([b, b_tmp])
    elif plan == 'jagged':
        b = []
        tmp_T = kwargs.pop('tmp_T')
        l_bd = kwargs.pop('lower_bd')
        u_bd = kwargs.pop('upper_bd')
        repetition = int(T / tmp_T) + 1
        b_tmp = np.arange(start=l_bd, stop=u_bd, step=(u_bd - l_bd) / tmp_T)
        for i in range(repetition):
            b = np.concatenate([b, (-1) ** i * b_tmp])
    elif plan == 'cos':
        b = []
        tmp_T = kwargs.pop('tmp_T')
        magnitude_cos = kwargs.pop('magnitude_cos')
        repetition = int(T / tmp_T) + 1
        # b_tmp = np.arange(start=l_bd, stop=u_bd, step=(u_bd - l_bd) / tmp_T)
        x_tmp = np.arange(start=0, stop=2 * np.pi, step=(2 * np.pi) / tmp_T)
        b_tmp = magnitude_cos * np.cos(x_tmp)
        for i in range(repetition):
            b = np.concatenate([b, (-1) ** i * b_tmp])
    elif plan == 'leap':
        b = []
        tmp_T = kwargs.pop('tmp_T')
        mag = kwargs.pop('mag')
        repetition = int(T / tmp_T) + 1
        b_tmp = np.ones(tmp_T) * mag
        for i in range(repetition):
            b = np.concatenate([b, (-1) ** i * b_tmp])
    elif plan == 'leap_b':
        b = []
        tmp_T = kwargs.pop('tmp_T')
        mag = kwargs.pop('mag')
        repetition = int(T / tmp_T) + 1
        b_tmp = np.ones(tmp_T) * mag
        for i in range(repetition):
            if i % 3 == 0:
                b = np.concatenate([b, -1 * b_tmp])
            else:
                b = np.concatenate([b, b_tmp])
    # plt.plot(b)
    return b[:T]


def func(a, x, b):
    return (a * x - b) ** 2 / 2


def grad(a, x, b):
    return a * (a * x - b)


def fun_opt(a, b, lower_bd, upper_bd):
    """
    optimal variable value and the optimal function value. This has closed form for quadratic functions.
    :param a:
    :param b:
    :param lower_bd:
    :param upper_bd:
    :return: the optimal objective function and the optimal variable.
    """
    opt_tmp = b / a if a != 0 else lower_bd
    if opt_tmp > upper_bd:
        return func(a, upper_bd, b), upper_bd
    elif opt_tmp < lower_bd:
        return func(a, lower_bd, b), lower_bd
    else:
        return 0, opt_tmp


def box_proj(xt, lower_bd, upper_bd):
    """
    Projection of box constrain set.
    :param xt:
    :param lower_bd:
    :param upper_bd:
    :return:
    """
    if xt < lower_bd:
        return lower_bd
    elif xt > upper_bd:
        return upper_bd
    else:
        return xt


def attack(grad: float, attack_type: str, attack_magnitude: int) -> np:
    """
    change the gradient when attack condition is satisfied.
    :param grad:
    :param attack_type:
    :param attack_magnitude:
    :return: the changed gradient
    """
    if attack_type == 'flipping':
        return - grad * attack_magnitude
    elif attack_type == 'zero':
        return np.zeros_like(grad)
    elif attack_type == 'random':  # notice the function is a R\to R, the parameter is a scalar.
        return (np.random.random()-0.5) * attack_magnitude * np.linalg.norm(grad)
        # return np.random.random(grad.shape) * attack_magnitude * np.linalg.norm(grad)
    else:
        raise Exception("attack type undefined.")


def plot_tmp(x_coll, avg_reg, fig_name, savefig=True):
    """
    plot for one
    :param fig_name:
    :param x_coll:
    :param avg_reg:
    :param savefig:
    :return:
    """
    x_coll_omgd = x_coll['x_coll_omgd']
    x_coll_opt = x_coll['x_coll_opt']
    x_coll_l3gd = x_coll['x_coll_l3gd']
    x_coll_ongd = x_coll['x_coll_ongd']

    avg_reg_omgd = avg_reg['avg_reg_omgd']
    avg_reg_l3gd = avg_reg['avg_reg_l3gd']
    avg_reg_ongd = avg_reg['avg_reg_ongd']

    plt.figure()
    fig, axs = plt.subplots(1, 2)  # , constrained_layout=True)
    axs[0].plot(x_coll_omgd[:-2], 'k')
    axs[0].plot(x_coll_opt[:-2], 'g')
    axs[0].plot(x_coll_l3gd[:-2], 'r')
    axs[0].plot(x_coll_ongd[:-2], 'b')

    # axs[1].plot(avg_reg_omgd, 'k')
    # axs[1].plot(avg_reg_l3gd, 'r')
    # axs[1].plot(avg_reg_ongd, 'b')

    axs[1].semilogy(avg_reg_omgd, 'k')
    axs[1].semilogy(avg_reg_l3gd, 'r')
    axs[1].semilogy(avg_reg_ongd, 'b')

    # axs[1].set_ylim([0, 500])

    if savefig:
        plt.savefig('../assets/syn_{}.png'.format(fig_name))
    plt.show()

def show_res_last(res_dict):
    for k, v in res_dict.items():
        print(k)
        print(v[-1])
        print('-------')




