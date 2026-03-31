#!/usr/bin/env python3

'''
Created on 7 Mar 2023

@author: steam
'''

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.scenarios import test_three_row_scenario
from common.airport_map_drawer import AirportMapDrawer

from monte_carlo.on_policy_mc_predictor import OnPolicyMCPredictor
from monte_carlo.off_policy_mc_predictor import OffPolicyMCPredictor

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer
from generalized_policy_iteration.policy_evaluator import PolicyEvaluator

from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
from p1.low_level_policy_drawer import LowLevelPolicyDrawer


def compute_rmse(v_true, v_est, airport_map):
    squared_errors = []

    for x in range(airport_map.width()):
        for y in range(airport_map.height()):
            cell = airport_map.cell(x, y)

            if cell.is_obstruction() or cell.is_terminal():
                continue

            true_val = v_true.value(x, y)
            est_val = v_est.value(x, y)

            if math.isnan(true_val) or math.isnan(est_val):
                continue

            error = est_val - true_val
            squared_errors.append(error ** 2)

    if len(squared_errors) == 0:
        return float("nan")

    return math.sqrt(sum(squared_errors) / len(squared_errors))


if __name__ == '__main__':
    airport_map, drawer_height = test_three_row_scenario()
    env = LowLevelEnvironment(airport_map)
    env.set_nominal_direction_probability(0.8)

    # Policy to evaluate
    pi = env.initial_policy()
    pi.set_epsilon(0)
    pi.set_action(14, 1, LowLevelActionType.MOVE_DOWN)
    pi.set_action(14, 2, LowLevelActionType.MOVE_DOWN)

    # Policy evaluation algorithm
    pe = PolicyEvaluator(env)
    pe.set_policy(pi)
    v_pe = ValueFunctionDrawer(pe.value_function(), drawer_height)
    pe.evaluate()
    v_pe.update()
    v_pe.update()

    # On-policy MC predictor: first-visit
    mcpp_first = OnPolicyMCPredictor(env)
    mcpp_first.set_target_policy(pi)
    mcpp_first.set_experience_replay_buffer_size(64)
    mcpp_first.set_use_first_visit(True)
    v_mcpp_first = ValueFunctionDrawer(mcpp_first.value_function(), drawer_height)

    # On-policy MC predictor: every-visit
    mcpp_every = OnPolicyMCPredictor(env)
    mcpp_every.set_target_policy(pi)
    mcpp_every.set_experience_replay_buffer_size(64)
    mcpp_every.set_use_first_visit(False)
    v_mcpp_every = ValueFunctionDrawer(mcpp_every.value_function(), drawer_height)

    # Off-policy MC predictor: first-visit
    mcop_first = OffPolicyMCPredictor(env)
    mcop_first.set_target_policy(pi)
    mcop_first.set_experience_replay_buffer_size(64)
    b_first = env.initial_policy()
    b_first.set_epsilon(0.2)
    mcop_first.set_behaviour_policy(b_first)
    mcop_first.set_use_first_visit(True)
    v_mcop_first = ValueFunctionDrawer(mcop_first.value_function(), drawer_height)

    # Off-policy MC predictor: every-visit
    mcop_every = OffPolicyMCPredictor(env)
    mcop_every.set_target_policy(pi)
    mcop_every.set_experience_replay_buffer_size(64)
    b_every = env.initial_policy()
    b_every.set_epsilon(0.2)
    mcop_every.set_behaviour_policy(b_every)
    mcop_every.set_use_first_visit(False)
    v_mcop_every = ValueFunctionDrawer(mcop_every.value_function(), drawer_height)

    episode_numbers = []
    rmse_on_first = []
    rmse_on_every = []
    rmse_off_first = []
    rmse_off_every = []

    for e in range(100):
        episode_numbers.append(e + 1)

        mcpp_first.evaluate()
        v_mcpp_first.update()
        rmse_on_first.append(
            compute_rmse(pe.value_function(), mcpp_first.value_function(), airport_map)
        )

        mcpp_every.evaluate()
        v_mcpp_every.update()
        rmse_on_every.append(
            compute_rmse(pe.value_function(), mcpp_every.value_function(), airport_map)
        )

        mcop_first.evaluate()
        v_mcop_first.update()
        rmse_off_first.append(
            compute_rmse(pe.value_function(), mcop_first.value_function(), airport_map)
        )

        mcop_every.evaluate()
        v_mcop_every.update()
        rmse_off_every.append(
            compute_rmse(pe.value_function(), mcop_every.value_function(), airport_map)
        )

    print(f"Final on-policy first-visit RMSE: {rmse_on_first[-1]:.4f}")
    print(f"Final on-policy every-visit RMSE: {rmse_on_every[-1]:.4f}")
    print(f"Final off-policy first-visit RMSE: {rmse_off_first[-1]:.4f}")
    print(f"Final off-policy every-visit RMSE: {rmse_off_every[-1]:.4f}")

    plt.figure()
    plt.plot(episode_numbers, rmse_on_first, label="On-policy first-visit")
    plt.plot(episode_numbers, rmse_on_every, label="On-policy every-visit")
    plt.plot(episode_numbers, rmse_off_first, label="Off-policy first-visit")
    plt.plot(episode_numbers, rmse_off_every, label="Off-policy every-visit")
    plt.xlabel("Evaluation step")
    plt.ylabel("RMSE")
    plt.title("Q1(b) RMSE Against Ground Truth")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("q1_b_rmse.png", dpi=200)
    plt.close()

    # Sample way to generate outputs
    v_pe.save_screenshot("q1_b_truth_pe.pdf")
    v_mcpp_first.save_screenshot("q1_b_mc-on_first-visit_pe.pdf")
    v_mcpp_every.save_screenshot("q1_b_mc-on_every-visit_pe.pdf")
    v_mcop_first.save_screenshot("q1_b_mc-off_first-visit_pe.pdf")
    v_mcop_every.save_screenshot("q1_b_mc-off_every-visit_pe.pdf")