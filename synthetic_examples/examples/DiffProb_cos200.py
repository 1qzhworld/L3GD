import pickle
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import sys
sys.path.append(os.path.abspath("."))
print(os.path.abspath("."))
from synthetic_examples.utilities.synthesis_main import main

start_time = datetime.now()
savefig = False
a_plan = 'ones'
b_plan_cos = {
    "plan": "cos",
    "tmp_T": 200,
    "magnitude_cos": 10
}
prob = {
    "T": 10000,
    "rand_seed": 9,
    "constrain": {
        "u_bd": 10,
        "l_bd": -10
    },
}
alg_setting_thm_dim_stp = {
    "eta": "thm_dim_stp",
    "xinit": 0,
    "Kt_type": "dim_thm"
}


attack_dict95 = {
    "attack_prob": 0.5,
    "attack_mag": 9,
    "attack_type": "flipping"
}
avg_reg_omgd_p05m9, avg_reg_l3gd_p05m9, avg_reg_ongd_p05m9, \
x_coll_opt_p05m9, x_coll_omgd_p05m9, x_coll_l3gd_p05m9, x_coll_ongd_p05m9 = \
    main(a_plan, b_plan_cos, attack_dict95, prob, alg_setting_thm_dim_stp)

attack_dict94 = {
    "attack_prob": 0.4,
    "attack_mag": 9,
    "attack_type": "flipping"
}
avg_reg_omgd_p04m9, avg_reg_l3gd_p04m9, avg_reg_ongd_p04m9, \
x_coll_opt_p04m9, x_coll_omgd_p04m9, x_coll_l3gd_p04m9, x_coll_ongd_p04m9 = \
    main(a_plan, b_plan_cos, attack_dict94, prob, alg_setting_thm_dim_stp)

attack_dict93 = {
    "attack_prob": 0.3,
    "attack_mag": 9,
    "attack_type": "flipping"
}
avg_reg_omgd_p03m9, avg_reg_l3gd_p03m9, avg_reg_ongd_p03m9, \
x_coll_opt_p03m9, x_coll_omgd_p03m9, x_coll_l3gd_p03m9, x_coll_ongd_p03m9 = \
    main(a_plan, b_plan_cos, attack_dict93, prob, alg_setting_thm_dim_stp)

attack_dict92 = {
    "attack_prob": 0.2,
    "attack_mag": 9,
    "attack_type": "flipping"
}
avg_reg_omgd_p02m9, avg_reg_l3gd_p02m9, avg_reg_ongd_p02m9, \
x_coll_opt_p02m9, x_coll_omgd_p02m9, x_coll_l3gd_p02m9, x_coll_ongd_p02m9 = \
    main(a_plan, b_plan_cos, attack_dict92, prob, alg_setting_thm_dim_stp)

attack_dict91 = {
    "attack_prob": 0.1,
    "attack_mag": 9,
    "attack_type": "flipping"
}
avg_reg_omgd_p01m9, avg_reg_l3gd_p01m9, avg_reg_ongd_p01m9, \
x_coll_opt_p01m9, x_coll_omgd_p01m9, x_coll_l3gd_p01m9, x_coll_ongd_p01m9 = \
    main(a_plan, b_plan_cos, attack_dict91, prob, alg_setting_thm_dim_stp)

print('DONE!')

# plot the results.
fig, ax = plt.subplots()
# thm diminishing stepsize (theoretical stepsize and theoretical Kt).
alpha = 1
ax.semilogy(avg_reg_omgd_p05m9, 'k', alpha=alpha, label='p=0.5')
ax.semilogy(avg_reg_ongd_p05m9, 'b', alpha=alpha, label='p=0.5')
ax.semilogy(avg_reg_l3gd_p05m9, 'r', alpha=alpha, label='p=0.5')
alpha = 0.8
ax.semilogy(avg_reg_omgd_p04m9, 'k', alpha=alpha, label='p=0.4')
ax.semilogy(avg_reg_ongd_p04m9, 'b', alpha=alpha, label='p=0.4')
ax.semilogy(avg_reg_l3gd_p04m9, 'r', alpha=alpha, label='p=0.4')
alpha = 0.6
ax.semilogy(avg_reg_omgd_p03m9, 'k', alpha=alpha, label='p=0.3')
ax.semilogy(avg_reg_ongd_p03m9, 'b', alpha=alpha, label='p=0.3')
ax.semilogy(avg_reg_l3gd_p03m9, 'r', alpha=alpha, label='p=0.3')
alpha = 0.4
ax.semilogy(avg_reg_omgd_p02m9, 'k', alpha=alpha, label='p=0.2')
ax.semilogy(avg_reg_ongd_p02m9, 'b', alpha=alpha, label='p=0.2')
ax.semilogy(avg_reg_l3gd_p02m9, 'r', alpha=alpha, label='p=0.2')
alpha = 0.2
ax.semilogy(avg_reg_omgd_p01m9, 'k', alpha=alpha, label='p=0.1')
ax.semilogy(avg_reg_ongd_p01m9, 'b', alpha=alpha, label='p=0.1')
ax.semilogy(avg_reg_l3gd_p01m9, 'r', alpha=alpha, label='p=0.1')
plt.xlim([0, prob['T']])
ax2 = ax.twinx()
ax2.plot(np.NaN, np.NaN, label='OMGD', c='black')
ax2.plot(np.NaN, np.NaN, label='ONGD', c='blue')
ax2.plot(np.NaN, np.NaN, label='L3GD', c='red')

ax2.get_yaxis().set_visible(False)
ax.legend(loc=1)
ax2.legend(loc=2)
# plt.legend()
# plt.tight_layout()
ax.set_xlabel(r'time (T)')
ax.set_ylabel(r'$EReg_T^d/T$')
savefig = True
fig_name = start_time.strftime('DiffProb_cos200_thm_dim_stp' + '_%Y%m%d_%H%M')
if savefig:
    print(os.getcwd())
    try:
        plt.savefig('./synthetic_examples/assets/results_syn_{}.png'.format(fig_name), dpi=300)
    except:
        plt.savefig('../../synthetic_examples/assets/results_syn_{}.png'.format(fig_name), dpi=300)
plt.show()


# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------

alg_setting = {
    "eta": 1e-2,
    "xinit": 0,
    "Kt_type": "cst_thm"
}
avg_reg_omgd_cst_p05m9, avg_reg_l3gd_cst_p05m9, avg_reg_ongd_cst_p05m9, \
x_coll_opt_cst_p05m9, x_coll_omgd_cst_p05m9, x_coll_l3gd_cst_p05m9, x_coll_ongd_cst_p05m9 = \
    main(a_plan, b_plan_cos, attack_dict95, prob, alg_setting_thm_dim_stp)

avg_reg_omgd_cst_p04m9, avg_reg_l3gd_cst_p04m9, avg_reg_ongd_cst_p04m9, \
x_coll_opt_cst_p04m9, x_coll_omgd_cst_p04m9, x_coll_l3gd_cst_p04m9, x_coll_ongd_cst_p04m9 = \
    main(a_plan, b_plan_cos, attack_dict94, prob, alg_setting_thm_dim_stp)

avg_reg_omgd_cst_p03m9, avg_reg_l3gd_cst_p03m9, avg_reg_ongd_cst_p03m9, \
x_coll_opt_cst_p03m9, x_coll_omgd_cst_p03m9, x_coll_l3gd_cst_p03m9, x_coll_ongd_cst_p03m9 = \
    main(a_plan, b_plan_cos, attack_dict93, prob, alg_setting_thm_dim_stp)

avg_reg_omgd_cst_p02m9, avg_reg_l3gd_cst_p02m9, avg_reg_ongd_cst_p02m9, \
x_coll_opt_cst_p02m9, x_coll_omgd_cst_p02m9, x_coll_l3gd_cst_p02m9, x_coll_ongd_cst_p02m9 = \
    main(a_plan, b_plan_cos, attack_dict92, prob, alg_setting_thm_dim_stp)

avg_reg_omgd_cst_p01m9, avg_reg_l3gd_cst_p01m9, avg_reg_ongd_cst_p01m9, \
x_coll_opt_cst_p01m9, x_coll_omgd_cst_p01m9, x_coll_l3gd_cst_p01m9, x_coll_ongd_cst_p01m9 = \
    main(a_plan, b_plan_cos, attack_dict91, prob, alg_setting_thm_dim_stp)
print("DONE!:")

# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------

plt.figure()
# plot the results.
fig, ax = plt.subplots()
alpha = 1
ax.semilogy(avg_reg_omgd_cst_p05m9, 'k', alpha=alpha, label='p=0.5')
ax.semilogy(avg_reg_ongd_cst_p05m9, 'b', alpha=alpha, label='p=0.5')
ax.semilogy(avg_reg_l3gd_cst_p05m9, 'r', alpha=alpha, label='p=0.5')
alpha = 0.8
ax.semilogy(avg_reg_omgd_cst_p04m9, 'k', alpha=alpha, label='p=0.4')
ax.semilogy(avg_reg_ongd_cst_p04m9, 'b', alpha=alpha, label='p=0.4')
ax.semilogy(avg_reg_l3gd_cst_p04m9, 'r', alpha=alpha, label='p=0.4')
alpha = 0.6
ax.semilogy(avg_reg_omgd_cst_p03m9, 'k', alpha=alpha, label='p=0.3')
ax.semilogy(avg_reg_ongd_cst_p03m9, 'b', alpha=alpha, label='p=0.3')
ax.semilogy(avg_reg_l3gd_cst_p03m9, 'r', alpha=alpha, label='p=0.3')
alpha = 0.4
ax.semilogy(avg_reg_omgd_cst_p02m9, 'k', alpha=alpha, label='p=0.2')
ax.semilogy(avg_reg_ongd_cst_p02m9, 'b', alpha=alpha, label='p=0.2')
ax.semilogy(avg_reg_l3gd_cst_p02m9, 'r', alpha=alpha, label='p=0.2')
alpha = 0.2
ax.semilogy(avg_reg_omgd_cst_p01m9, 'k', alpha=alpha, label='p=0.1')
ax.semilogy(avg_reg_ongd_cst_p01m9, 'b', alpha=alpha, label='p=0.1')
ax.semilogy(avg_reg_l3gd_cst_p01m9, 'r', alpha=alpha, label='p=0.1')
plt.xlim([0, prob['T']])
ax2 = ax.twinx()
ax2.plot(np.NaN, np.NaN, label='OMGD', c='black')
ax2.plot(np.NaN, np.NaN, label='ONGD', c='blue')
ax2.plot(np.NaN, np.NaN, label='L3GD', c='red')

ax2.get_yaxis().set_visible(False)
ax.legend(loc=1)
ax2.legend(loc=2)
# plt.legend()
# plt.tight_layout()
ax.set_xlabel(r'time (T)')
ax.set_ylabel(r'$Reg_T^d/T$')
savefig = True
fig_name = start_time.strftime('DiffProb_cos200_cst_stp' + '_%Y%m%d_%H%M')
if savefig:
    print(os.getcwd())
    try:
        plt.savefig('./synthetic_examples/assets/results_syn_{}.png'.format(fig_name), dpi=300)
    except:
        plt.savefig('../../synthetic_examples/assets/results_syn_{}.png'.format(fig_name), dpi=300)
plt.show()

# --------------------------------------------------
# --------------For comparisons-----------------
# --------------------------------------------------

plt.figure()
# plot the results.
fig, ax = plt.subplots()
alpha = 1
ax.semilogy(avg_reg_l3gd_cst_p05m9, 'r', alpha=alpha, label='p=0.5')
ax.semilogy(avg_reg_l3gd_p05m9, 'g', alpha=alpha, label='p=0.5')
alpha = 0.8
ax.semilogy(avg_reg_l3gd_cst_p04m9, 'r', alpha=alpha, label='p=0.4')
ax.semilogy(avg_reg_l3gd_p04m9, 'g', alpha=alpha, label='p=0.4')
alpha = 0.6
ax.semilogy(avg_reg_l3gd_cst_p03m9, 'r', alpha=alpha, label='p=0.3')
ax.semilogy(avg_reg_l3gd_p03m9, 'g', alpha=alpha, label='p=0.3')
alpha = 0.4
ax.semilogy(avg_reg_l3gd_cst_p02m9, 'r', alpha=alpha, label='p=0.2')
ax.semilogy(avg_reg_l3gd_p02m9, 'g', alpha=alpha, label='p=0.2')
alpha = 0.2
ax.semilogy(avg_reg_l3gd_cst_p01m9, 'r', alpha=alpha, label='p=0.1')
ax.semilogy(avg_reg_l3gd_p01m9, 'g', alpha=alpha, label='p=0.1')
plt.xlim([0, prob['T']])
ax2 = ax.twinx()

ax2.plot(np.NaN, np.NaN, label='L3GD with a constant stepsize', c='red')
ax2.plot(np.NaN, np.NaN, label='L3GD with a diminishing stepsize', c='green')

ax2.get_yaxis().set_visible(False)
ax.legend(loc=1)
ax2.legend(loc=2)

ax.set_xlabel(r'time (T)')
ax.set_ylabel(r'$EReg_T^d/T$')
savefig = True
fig_name = start_time.strftime('DiffProb_cos200_compare' + '_%Y%m%d_%H%M')
if savefig:
    print(os.getcwd())
    try:
        plt.savefig('./synthetic_examples/assets/results_syn_{}.png'.format(fig_name), dpi=300)
    except:
        plt.savefig('../../synthetic_examples/assets/results_syn_{}.png'.format(fig_name), dpi=300)
plt.show()

res_dict = {}
res_keys = []
local_keys = list(locals().keys())
for k in local_keys:
    if 'avg_reg_' in k:
        res_keys.append(k)
for res_key in res_keys:
    res_dict[res_key] = locals()[res_key]

try:
    with open('./synthetic_examples/assets/results_syn_{}.pickle'.format(fig_name), 'wb') as handle:
        pickle.dump(res_dict, handle)
except:
    with open('../../synthetic_examples/assets/results_syn_{}.pickle'.format(fig_name), 'wb') as handle:
        pickle.dump(res_dict, handle)

# with open('./synthetic_examples/assets/results_syn_{}.pickle'.format(fig_name), 'rb') as handle:
#     res_dict = pickle.load(handle)


# for k, v in res_dict.items():
#     print(k)
#     print(v[-1])
#     print('-------')
