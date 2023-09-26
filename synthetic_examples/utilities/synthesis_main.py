import numpy as np
from scipy.special import lambertw
import os
import sys
sys.path.append(os.path.abspath("."))
# import local functions
from synthetic_examples.utilities.synthesis_utilities import generate_opt, attack, grad, func, fun_opt, plot_tmp
from synthetic_examples.utilities.opt_algs import grad_desc, nor_grad_desc


def main(a_plan, b_plan, attack_dict, prob, alg_setting):

    # problem related parameter setting
    T = prob['T']
    rand_seed = prob['rand_seed']  # this is used for attack prob.
    l_bd, u_bd = prob['constrain']['l_bd'], prob['constrain']['u_bd']
    assert l_bd < u_bd

    # attack setting
    attack_mag = attack_dict['attack_mag']
    attack_prob = attack_dict['attack_prob']
    attack_type = attack_dict['attack_type']

    # algorithm setting
    if type(alg_setting['eta']) == float:
        eta = alg_setting['eta']
    elif type(alg_setting['eta']) == str:
        if alg_setting['eta'] == 'thm_dim_stp':
            D = u_bd-l_bd
            cosphi = 1
            q = attack_prob
            if q >= cosphi/(1+cosphi):
                q = attack_prob-0.1  # this is for simulation setting, theoretically if q<p the alg will fail.

            eta = D / (2*((1-q)*cosphi-q))
            # theoretical diminishing stepsize should show up with Kt with dim_thm.
            assert alg_setting['Kt_type'] == 'dim_thm'

        elif alg_setting['eta'] == 'thm_cst_stp':
            # TODO:
            assert alg_setting['Kt_type'] == 'cst_thm'
            pass

    xinit = alg_setting['xinit']
    Kt_type = alg_setting['Kt_type']

    # generate a_plan
    if a_plan == 'ones':
        a = np.ones(T)
    # generate b_plan
    # As different generation plan requires different parameters, this is set when call.
    # generate b_plan
    if b_plan['plan'] == 'jagged':
        b = generate_opt(plan='jagged', T=T,
                         tmp_T=b_plan['tmp_T'], lower_bd=b_plan['l_bd'], upper_bd=b_plan['u_bd'])
    elif b_plan['plan'] == 'random':
        b = generate_opt(plan='random', T=T, rand_seed=b_plan['rand_seed'],
                         upper_rand=b_plan['upper_rand'], lower_rand=b_plan['lower_rand'])
    elif b_plan['plan'] == 'leap':
        b = generate_opt(plan="leap", T=T,
                         tmp_T=b_plan['tmp_T'], mag=b_plan['opt_mag'])
    elif b_plan['plan'] == 'cos':
        b = generate_opt(plan='cos', T=T,
                         tmp_T=b_plan['tmp_T'], magnitude_cos=b_plan['magnitude_cos'])

    eta_omgd, eta_l3gd, eta_ongd = eta, eta, eta
    xt_omgd, xt_l3gd, xt_ongd = xinit, xinit, xinit

    # collection initialization
    reg_omgd, reg_l3gd, reg_ongd = [0], [0], [0]
    x_coll_omgd, x_coll_l3gd, x_coll_ongd = \
        np.zeros(T), np.zeros(T), np.zeros(T)
    x_coll_opt = np.zeros(T)

    # objective function
    # ---------------------------------------------------------------------------------------------------
    # ------------------------------------------ main ---------------------------------------------------
    # ---------------------------------------------------------------------------------------------------
    for t in range(T - 1):
        if Kt_type == "outer_count":
            Kt = t + 1
        elif Kt_type == "cst_thm":  # this is used, ignore the others first.
            Kt = int(np.sqrt(t + 1) * np.log(t + 1))
        elif Kt_type == "dim_thm":
            Kt_tmp = (-(t**0.5)*lambertw(-(1/(np.exp(1)*(t**0.5))), k=-1)).real
            if Kt_tmp > 1:
                Kt = int(Kt_tmp) + 1
                # print(Kt)
            else:
                Kt = t
                print("Kt_tmp <= 1, use Kt=t")
            # Kt = Kt*(D**2)
            # if t % 100 == 0:
            #    print("t: {}, Kt: {}".format(t,Kt))
            # print(Kt)
        assert ('Kt' in locals())


        r = 0  # used by xt_ongd
        # r has different length for different t, so is not given in the parameter, but set during running.
        # run different algorithms
        for kt in range(Kt):
            # if attack
            r = np.random.rand()  # attack or not.
            grad_omgd = grad(a[t], xt_omgd, b[t]) if r > attack_prob \
                else attack(grad(a[t], xt_omgd, b[t]), attack_type, attack_mag)
            grad_l3gd = grad(a[t], xt_l3gd, b[t]) if r > attack_prob \
                else attack(grad(a[t], xt_l3gd, b[t]), attack_type, attack_mag)
            if alg_setting['eta'] == 'thm_dim_stp':
                eta_omgd, eta_l3gd = eta/(kt+1), eta/(kt+1)
                # print(eta_l3gd)
            xt_omgd = grad_desc(xt_omgd, eta_omgd, grad_omgd, l_bd, u_bd)
            xt_l3gd = nor_grad_desc(xt_l3gd, eta_l3gd, grad_l3gd, l_bd, u_bd)

        grad_ongd = grad(a[t], xt_ongd, b[t]) if r > attack_prob \
            else attack(grad(a[t], xt_ongd, b[t]), attack_type, attack_mag)
        xt_ongd = nor_grad_desc(xt_ongd, eta_ongd, grad_ongd, l_bd, u_bd)

        # calculate the regret
        next_t = t + 1
        ft_opt, xt_opt = fun_opt(a[next_t], b[next_t], l_bd, u_bd)

        reg_omgd.append(reg_omgd[-1] + func(a[next_t], xt_omgd, b[next_t]) - ft_opt)
        reg_l3gd.append(reg_l3gd[-1] + func(a[next_t], xt_l3gd, b[next_t]) - ft_opt)
        reg_ongd.append(reg_ongd[-1] + func(a[next_t], xt_ongd, b[next_t]) - ft_opt)

        x_coll_opt[t] = xt_opt
        x_coll_omgd[t] = xt_omgd
        x_coll_l3gd[t] = xt_l3gd
        x_coll_ongd[t] = xt_ongd

    # averaged regret (divided t), the first should be ignored.
    avg_reg_omgd = [reg / (ind + 1) for ind, reg in enumerate(reg_omgd[1:])]
    avg_reg_l3gd = [reg / (ind + 1) for ind, reg in enumerate(reg_l3gd[1:])]
    avg_reg_ongd = [reg / (ind + 1) for ind, reg in enumerate(reg_ongd[1:])]
    return avg_reg_omgd, avg_reg_l3gd, avg_reg_ongd, x_coll_opt, x_coll_omgd, x_coll_l3gd, x_coll_ongd


if __name__ == '__main__':
    a_plan = 'ones'
    b_plan = {
        "plan": "cos",
        "tmp_T": 200,
        "magnitude_cos": 10
    }
    prob = {
        "T": 1000,
        "rand_seed": 9,
        "constrain": {
            "u_bd": 10,
            "l_bd": -10
        },

    }
    alg_setting = {
        "eta": "thm_dim_stp",
        "xinit": 0,
        "Kt_type": "dim_thm"
    }
    attack_dict95 = {
        "attack_prob": 0.5,
        "attack_mag": 9,
        "attack_type": "flipping"
    }
    # avg_reg_omgd_p05m9, avg_reg_l3gd_p05m9, avg_reg_ongd_p05m9, \
    # x_coll_opt_p05m9, x_coll_omgd_p05m9, x_coll_l3gd_p05m9, x_coll_ongd_p05m9 = \
    #     all_algs_main(a_plan, b_plan, attack_dict95, prob, alg_setting)
    avg_reg_omgd, avg_reg_l3gd, avg_reg_ongd, x_coll_opt, x_coll_omgd, x_coll_l3gd, x_coll_ongd = \
        main(a_plan, b_plan, attack_dict95, prob, alg_setting)
    x_coll = {
        "x_coll_opt": x_coll_opt,
        "x_coll_omgd": x_coll_omgd,
        "x_coll_l3gd": x_coll_l3gd,
        "x_coll_ongd": x_coll_ongd
    }
    avg_reg = {
        "avg_reg_omgd": avg_reg_omgd,
        "avg_reg_l3gd": avg_reg_l3gd,
        "avg_reg_ongd": avg_reg_ongd
    }
    fig_name, savefig = 'test95', False
    plot_tmp(x_coll, avg_reg, fig_name, savefig=savefig)

