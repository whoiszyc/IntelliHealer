from statistics import mean
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque
import os
import csv
import numpy as np
from sys import exit

# SCORES_CSV_PATH = "/Users/whoiszyc/Github/gym-power-res/gym_power_res/sorces/scores.csv"
# SCORES_PNG_PATH = "/Users/whoiszyc/Github/gym-power-res/gym_power_res/sorces/scores.png"
# SOLVED_CSV_PATH = "/Users/whoiszyc/Github/gym-power-res/gym_power_res/sorces/solved.csv"
# SOLVED_PNG_PATH = "/Users/whoiszyc/Github/gym-power-res/gym_power_res/sorces/solved.png"


# The ideal action is [37] and [37, 35]
# The ideal reward is [10, 22, 31, 31, 31]=125

class ScoreLogger:
    line_colors = ["b", "r", "magenta", "lime", "darkorange", "cyan", "y", "purple", "deepskyblue", "navy", "salmon"]

    def __init__(self, env_name, output_path, AVERAGE_SCORE_TO_SOLVE, CONSECUTIVE_RUNS_TO_SOLVE, title=None):
        self.AVERAGE_SCORE_TO_SOLVE = AVERAGE_SCORE_TO_SOLVE
        self.CONSECUTIVE_RUNS_TO_SOLVE = CONSECUTIVE_RUNS_TO_SOLVE
        self.scores = deque(maxlen=self.CONSECUTIVE_RUNS_TO_SOLVE)
        self.env_name = env_name
        self.title = title

        # if os.path.exists(SCORES_PNG_PATH):
        #     os.remove(SCORES_PNG_PATH)
        # if os.path.exists(SCORES_CSV_PATH):
        #     os.remove(SCORES_CSV_PATH)
        now = datetime.now()
        dt_string = now.strftime("__%Y_%m_%d_%H_%M")

        self.SCORES_CSV_PATH = output_path + "/scores" + dt_string + ".csv"
        self.SCORES_PNG_PATH = output_path + "/scores" + dt_string + ".png"

    def add_score(self, score, run):
        self._save_csv(self.SCORES_CSV_PATH, score)
        self._save_png(input_path=self.SCORES_CSV_PATH,
                       output_path=self.SCORES_PNG_PATH,
                       x_label="runs",
                       y_label="restored load",
                       average_of_n_last=self.CONSECUTIVE_RUNS_TO_SOLVE,
                       show_goal=True,
                       show_trend=True,
                       show_legend=True)
        self.scores.append(score)
        flag_convergence = False
        mean_score = mean(self.scores)
        # print("Scores: (min: " + str(min(self.scores)) + ", avg: " + str(mean_score) + ", max: " + str(max(self.scores)) + ")\n")
        # Termination condition
        if mean_score >= self.AVERAGE_SCORE_TO_SOLVE and len(self.scores) >= self.CONSECUTIVE_RUNS_TO_SOLVE:
            solve_score = run - self.CONSECUTIVE_RUNS_TO_SOLVE
            print("Solved in " + str(solve_score) + " runs, " + str(run) + " total runs.")
            flag_convergence = True
        return flag_convergence

    def _save_png(self, input_path, output_path, x_label, y_label, average_of_n_last, show_goal, show_trend, show_legend):
        x = []
        y = []
        with open(input_path, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            for i in range(0, len(data)):
                x.append(int(i))
                y.append(int(data[i][0]))
        color_idx = 0
        plt.subplots()
        plt.plot(x, y, label="score per run", color=self.line_colors[color_idx])
        color_idx = color_idx + 1

        average_range = average_of_n_last if average_of_n_last is not None else len(x)
        plt.plot(x[-average_range:], [np.mean(y[-average_range:])] * len(y[-average_range:]),
                 linestyle="--", label="last " + str(average_range) + " runs average", color=self.line_colors[color_idx])
        color_idx = color_idx + 1

        if show_goal:
            plt.plot(x, [self.AVERAGE_SCORE_TO_SOLVE] * len(x),
                     linestyle=":", label=str(self.AVERAGE_SCORE_TO_SOLVE) + " score average goal", color=self.line_colors[color_idx])
        color_idx = color_idx + 1

        if show_trend and len(x) > 1:
            trend_x = x[1:]
            z = np.polyfit(np.array(trend_x), np.array(y[1:]), 1)
            p = np.poly1d(z)
            plt.plot(trend_x, p(trend_x), linestyle="-.",  label="trend", color=self.line_colors[color_idx])
        color_idx = color_idx + 1

        if self.title == None:
            title = self.env_name
        else:
            title = self.title
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if show_legend:
            plt.legend(loc="upper left")

        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def _save_csv(self, path, score):
        if not os.path.exists(path):
            with open(path, "w"):
                pass
        scores_file = open(path, "a")
        with scores_file:
            writer = csv.writer(scores_file)
            writer.writerow([score])



class SuccessLogger:
    line_colors = ["b", "r", "magenta", "lime", "darkorange", "cyan", "y", "purple", "deepskyblue", "navy", "salmon"]

    def __init__(self, env_name, output_path, title=None):
        self.success_ratio = []
        self.env_name = env_name
        self.title = title

        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        now = datetime.now()
        dt_string = now.strftime("__%Y_%m_%d_%H_%M")

        self.success_value_path = output_path + "/success_value" + dt_string + ".csv"
        self.success_room_path = output_path + "/success_room" + dt_string + ".csv"
        self.success_ratio_path = output_path + "/success_ratio" + dt_string + ".csv"
        self.success_ratio_1_path = output_path + "/success_ratio_1" + dt_string + ".csv"
        self.disturb_path = output_path + "/disturb" + dt_string + ".csv"
        self.SCORES_PNG_PATH = output_path + "/success_ratio" + dt_string + ".png"


    def add_score(self, performance_at_episode, goal_at_episode, value_at_episode, disturb_at_espisode, goal_1=None):
        if abs(goal_at_episode) < 1e-4:
            ratio = None
        else:
            ratio = performance_at_episode/goal_at_episode
        if goal_1 is not None:
            if abs(goal_1) < 1e-4:
                ratio_1 = None
            else:
                ratio_1 = performance_at_episode / goal_1
        room_at_episode = goal_at_episode - performance_at_episode
        self._save_csv(self.disturb_path, disturb_at_espisode)
        self._save_csv(self.success_value_path, value_at_episode)
        self._save_csv(self.success_room_path, room_at_episode)
        self._save_csv(self.success_ratio_path, ratio)
        if goal_1 is not None:
            self._save_csv(self.success_ratio_1_path, ratio_1)
        # self._save_png(input_path=self.success_ratio_path,
        #                output_path=self.SCORES_PNG_PATH,
        #                x_label="runs",
        #                y_label="restored load ratio",
        #                show_trend=False,
        #                show_legend=True)
        self.success_ratio.append(ratio)

    def _save_png(self, input_path, output_path, x_label, y_label, show_trend, show_legend):
        x = []
        y = []
        with open(input_path, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            for i in range(0, len(data)):
                x.append(int(i))
                try:
                    y.append(float(data[i][0]))
                except:
                    y.append(1)
        color_idx = 0
        plt.subplots()
        plt.plot(x, y, label="score per run", color=self.line_colors[color_idx])
        color_idx = color_idx + 1

        if show_trend and len(x) > 1:
            trend_x = x[1:]
            z = np.polyfit(np.array(trend_x), np.array(y[1:]), 1)
            p = np.poly1d(z)
            plt.plot(trend_x, p(trend_x), linestyle="-.",  label="trend", color=self.line_colors[color_idx])
        color_idx = color_idx + 1

        if self.title == None:
            title = self.env_name
        else:
            title = self.title
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if show_legend:
            plt.legend(loc="upper left")

        plt.savefig(output_path, bbox_inches="tight")
        plt.close()


    def _save_csv(self, path, score):
        if not os.path.exists(path):
            with open(path, "w"):
                pass
        scores_file = open(path, "a")
        with scores_file:
            writer = csv.writer(scores_file)
            writer.writerow([score])