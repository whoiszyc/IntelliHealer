import sys
import os
import time
import pandas as pd
import numpy as np
import math
import sympy as sy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import networkx as nx
import operator
from collections import OrderedDict
import pyomo.environ as pm


class SolutionDict(OrderedDict):
    """
    Solution dictionary is an ordered dictionary that stores the optimization results
    Solution dictionary struture: D[name key]=time series data
    """

    def copy_data(self, DictData):
        key_list = []
        for i in DictData.keys():
            key_list.append(i)


    def plot_2d(self, x_str='Time', y_str='Value', title_str='Results', figsize=(15,7), legendlist=None, show=True, save=False, folder=None):
        """step plot"""
        fig = plt.figure(figsize=figsize)

        key_list = []
        for i in self.keys():
            key_list.append(i)

        ## assume that all variables have the same time vector
        total_time = len(self[key_list[0]])

        if legendlist==None:
            legendlist = ['variable {}'.format(i) for i in key_list]

        k = 0
        for i in key_list:
            plt.plot(range(0, total_time), self[i], label=legendlist[k], linewidth=3)
            k = k + 1
        plt.title(title_str)
        plt.xlabel(x_str)
        plt.ylabel(y_str)
        plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
        if show==True:
            plt.show()
        if save==True:
            fig.savefig(folder + title_str)


    def plot_step_2d(self, x_str='Time', y_str='Value', title_str='Results', figsize=(15,7), legendlist=None, show=True, save=False, folder=None):
        """step plot"""
        fig = plt.figure(figsize=figsize)

        key_list = []
        for i in self.keys():
            key_list.append(i)

        ## assume that all variables have the same time vector
        total_time = len(self[key_list[0]])

        if legendlist==None:
            legendlist = ['variable {}'.format(i) for i in key_list]

        k = 0
        for i in key_list:
            plt.step(range(0, total_time), self[i], label=legendlist[k], linewidth=3)
            k = k + 1
        plt.title(title_str)
        plt.xlabel(x_str)
        plt.ylabel(y_str)
        plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
        if show==True:
            plt.show()
        if save==True:
            fig.savefig(folder + title_str)


    def plot_bin_2d(self, x_str='Time', y_str='Value', title_str='Results', figsize=(15,7), show=True, save=False, folder=None):
        """step plot"""
        fig = plt.figure(figsize=figsize)

        ## get key list
        key_list = []
        for i in self.keys():
            key_list.append(i)

        ## assume that all variables have the same time vector
        total_time = len(self[key_list[0]])

        y_axis = np.arange(0, len(key_list))
        k = 0
        for i in key_list:
            for t in range(0, total_time):
                if abs(self[i][t]) <= 0.01:
                    plt.scatter(t, y_axis[k], c='red', s=50, alpha=0.5, edgecolors='none')
                else:
                    plt.scatter(t, y_axis[k], c='green', s=50, alpha=0.5, edgecolors='none')
            k = k + 1
        plt.yticks(y_axis, key_list)
        plt.title(title_str)
        plt.xlabel(x_str)
        plt.ylabel(y_str)
        if show==True:
            plt.show()
        if save==True:
            fig.savefig(folder + title_str)



class OutageManage:
    """
    Distribution system restoration class
    """

    ## Optimization specification
    BigM = 100
    FicFlowBigM = 100
    epsilon = 0.1
    BasePower = 1000

    ## prepare optimization data
    def data_preparation(self, ppc, line_damaged, VS=1.05, dV=0.05):
        """
        read data of distribution network and vehicle routing problem
        Data is in the format of dictionary
        """
        self.VS = VS
        self.dV = dV

        self.ppc = ppc

        ## get distribution network data
        self.number_bus = ppc['number_bus']
        self.number_line = ppc['number_line']
        self.number_gen = ppc['number_gen']
        self.index_bus = ppc['index_bus']
        self.index_line = ppc['index_line']
        self.index_gen = ppc['index_gen']
        self.iter_bus = ppc['iter_bus']
        self.iter_gen = ppc['iter_gen']
        self.iter_line = ppc['iter_line']
        self.bus_line = ppc['bus_line']
        self.bus_gen = ppc['bus_gen']

        ## sort line type
        # (1)fixed line, (2)damaged line, (3)tie line
        self.line_damaged = line_damaged
        self.line_switch = ppc['tieline']
        self.line_static = set(self.iter_line) - set(self.line_switch) - set(self.line_damaged)


    def solve_network_restoration_varcon(self, Total_Step, Line_Initial, VarCon_Initial, Load_Initial):
        """
        solve a full-step restoration problem with reactive power dispatch
        :param Total_Step: total step of the problem
        :param Line_Initial: initial line status
        :return: restoration plan
        """
        self.model = pm.ConcreteModel()
        self.total_step = Total_Step
        self.iter_time = np.arange(0, Total_Step)
        ppc = self.ppc

        ## ================== network operation variables ====================
        self.model.p = pm.Var(self.iter_gen, self.iter_time, within=pm.Reals)
        self.model.q = pm.Var(self.iter_gen, self.iter_time, within=pm.Reals)
        self.model.P = pm.Var(self.iter_line, self.iter_time, within=pm.Reals)
        self.model.Q = pm.Var(self.iter_line, self.iter_time, within=pm.Reals)
        self.model.V = pm.Var(self.iter_bus, self.iter_time, within=pm.NonNegativeReals)
        self.model.rho = pm.Var(self.iter_bus, self.iter_time, within=pm.Binary)
        self.model.ug = pm.Var(self.iter_gen, self.iter_time, within=pm.Binary)
        self.model.ul = pm.Var(self.iter_line, self.iter_time, within=pm.Binary)
        self.model.beta = pm.Var(self.iter_bus, self.iter_bus, self.iter_time, within=pm.Binary)
        self.model.nb = pm.Var(self.iter_time, within=pm.NonNegativeReals)

        ## =================== distflow constraint: power balance at bus i ===================
        self.model.con_distflow_bus = pm.ConstraintList()  # In pyomo formulation, use ConstraintList() is a generic way to add constraints
        for t in self.iter_time:
            for i in self.iter_bus:
                P_out = 0
                Q_out = 0
                P_in = 0
                Q_in = 0

                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:])
                idx = np.where(id == ppc['bus'][:, 0])[0]
                idx = idx[0]  # the original idx was an array, since bus and line idx is unique, we get the entry

                # get power flow variables flowing out from this bus
                iter_out = np.arange(0, len(self.bus_line[i]["line_from_this_bus"]))

                if iter_out.shape[0] == 0:
                    P_out = P_out
                    Q_out = Q_out
                else:
                    for k in iter_out:
                        _id_from = int(self.bus_line[i]["line_from_this_bus"][k])
                        P_out = P_out + self.model.P['line_{}'.format(_id_from), t]
                        Q_out = Q_out + self.model.Q['line_{}'.format(_id_from), t]

                # get power flow variables flowing into from this bus
                iter_in = np.arange(0, len(self.bus_line[i]["line_to_this_bus"]))

                if iter_in.shape[0] == 0:
                    P_in = P_in
                    Q_in = Q_in
                else:
                    for k in iter_in:
                        _id_to = int(self.bus_line[i]["line_to_this_bus"][k])
                        P_in = P_in + self.model.P['line_{}'.format(_id_to), t]
                        Q_in = Q_in + self.model.Q['line_{}'.format(_id_to), t]

                # if this is a bus with generation capability and load, the power flow balance is
                if ppc['bus'][idx, 1] == 1:
                    self.model.con_distflow_bus.add(
                        P_in + self.model.p['gen_{}'.format(int(self.bus_gen[i][0])), t] == P_out + ppc['bus'][idx, 4] * self.model.rho[i, t])
                    self.model.con_distflow_bus.add(
                        Q_in + self.model.q['gen_{}'.format(int(self.bus_gen[i][0])), t] == Q_out + ppc['bus'][idx, 5] * self.model.rho[i, t])

                # if this is a bus with only load
                else:
                    self.model.con_distflow_bus.add(P_in == P_out + ppc['bus'][idx, 4] * self.model.rho[i, t])
                    self.model.con_distflow_bus.add(Q_in == Q_out + ppc['bus'][idx, 5] * self.model.rho[i, t])

        ## ===================  distflow constraint: voltage drop along line k ===================
        self.model.con_distflow_line = pm.ConstraintList()
        for t in self.iter_time:
            for i in self.iter_line:
                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:])
                idx = np.where(id == ppc['line'][:, 0])[0]
                idx = idx[0]  # the original idx was an array, since bus and line idx is unique, we get the entry

                # for this line, get the bus index
                f_bus = int(ppc['line'][idx, 1])
                t_bus = int(ppc['line'][idx, 2])

                self.model.con_distflow_line.add(
                    self.model.V['bus_{}'.format(t_bus), t] - self.model.V['bus_{}'.format(f_bus), t]
                    + (ppc['line'][idx, 3] * self.model.P[i, t] + ppc['line'][idx, 4] * self.model.Q[i, t])/self.VS <= (1 - self.model.ul[i, t]) * self.BigM)
                self.model.con_distflow_line.add(
                    self.model.V['bus_{}'.format(t_bus), t] - self.model.V['bus_{}'.format(f_bus), t]
                    + (ppc['line'][idx, 3] * self.model.P[i, t] + ppc['line'][idx, 4] * self.model.Q[i, t])/self.VS >= -(1 - self.model.ul[i, t]) * self.BigM)

        # ======================== Power flow limits =========================
        self.model.con_lim_line = pm.ConstraintList()
        for t in self.iter_time:
            for i in self.iter_line:
                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:])
                idx = np.where(id == ppc['line'][:, 0])[0]
                idx = idx[0]  # the original idx was an array, since bus and line idx is unique, we get the entry
                self.model.con_lim_line.add(self.model.P[i, t] <= self.model.ul[i, t] * ppc['line'][idx, 5])  #ppc['line'][id, 5]
                self.model.con_lim_line.add(self.model.P[i, t] >= -self.model.ul[i, t] * ppc['line'][idx, 5])
                self.model.con_lim_line.add(self.model.Q[i, t] <= self.model.ul[i, t] * ppc['line'][idx, 6]) # ppc['line'][id, 6]
                self.model.con_lim_line.add(self.model.Q[i, t] >= -self.model.ul[i, t] * ppc['line'][idx, 6])

        #  ======================== voltage deviation limits =========================
        self.model.con_lim_voltage = pm.ConstraintList()
        for t in self.iter_time:
            for i in self.iter_bus:
                if i == 'bus_1':
                    self.model.con_lim_voltage.add(self.model.V[i, t] == self.VS)
                else:
                    self.model.con_lim_voltage.add(self.model.V[i, t] >= 1 - self.dV)
                    self.model.con_lim_voltage.add(self.model.V[i, t] <= 1 + self.dV)

        ## ===================== tree topology constraints ==========================
        self.model.con_radiality = pm.ConstraintList()

        # substation bus does not have parent bus
        for t in self.iter_time:
            self.model.con_radiality.add(self.model.beta['bus_2', 'bus_1', t] == 0)  # hard coded

        # each bus will only at least have one parent bus
        for t in self.iter_time:
            for j in self.iter_bus:
                self.model.con_radiality.add(sum(self.model.beta[i, j, t] for i in self.iter_bus) <= 1)

        for t in self.iter_time:
            for i in self.iter_line:
                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:])
                idx = np.where(id == ppc['line'][:, 0])[0]
                idx = idx[0]  # the original idx was an array, since bus and line idx is unique, we get the entry
                # for this line, get the bus index
                f_bus = int(ppc['line'][idx, 1])
                t_bus = int(ppc['line'][idx, 2])
                self.model.con_radiality.add(self.model.beta['bus_{}'.format(f_bus), 'bus_{}'.format(t_bus), t]
                                        + self.model.beta['bus_{}'.format(t_bus), 'bus_{}'.format(f_bus), t] == self.model.ul[i, t])

        # ================== generator dispatch ===============
        # generator dispatch limits
        self.model.con_lim_gen = pm.ConstraintList()
        for t in self.iter_time:
            for i in self.iter_gen:
                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:])
                idx = np.where(id == ppc['gen'][:, 0])[0]
                idx = idx[0]  # the original idx was an array, since bus and line idx is unique, we get the entry
                self.model.con_lim_gen.add(self.model.p[i, t] >= ppc['gen'][idx, 2] * self.model.ug[i, t])
                self.model.con_lim_gen.add(self.model.p[i, t] <= ppc['gen'][idx, 3] * self.model.ug[i, t])
                self.model.con_lim_gen.add(self.model.q[i, t] >= ppc['gen'][idx, 4] * self.model.ug[i, t])
                self.model.con_lim_gen.add(self.model.q[i, t] <= ppc['gen'][idx, 5] * self.model.ug[i, t])

        # enforce initial conditions of dispatchable generators to be zero
        for i in VarCon_Initial:
            self.model.con_lim_gen.add(self.model.q[i, 0] == VarCon_Initial[i])

        # minimize the dispatch frequencies
        # z = 1 indicates that self.model.q[i, t] - self.model.q[i, t - 1] is positive
        # Dq(t) = |q(t) - q(t-1)|
        self.model.con_dispatch_freq = pm.ConstraintList()
        self.model.z = pm.Var(self.ppc['varcon'], self.iter_time, within=pm.Binary)
        self.model.Dq = pm.Var(self.ppc['varcon'], self.iter_time, within=pm.NonNegativeReals)
        for t in range(1, self.total_step):
            for i in self.ppc['varcon']:
                self.model.con_dispatch_freq.add(self.model.q[i, t] - self.model.q[i, t - 1] >= (1 - self.model.z[i, t]) * (-1000))
                self.model.con_dispatch_freq.add(-1000 * (1 - self.model.z[i, t]) <= self.model.Dq[i, t] - (self.model.q[i, t] - self.model.q[i, t-1]))
                self.model.con_dispatch_freq.add(self.model.Dq[i, t] - (self.model.q[i, t] - self.model.q[i, t - 1]) <= (1 - self.model.z[i, t]) * 1000)
                self.model.con_dispatch_freq.add(-1000 * self.model.z[i, t] <= self.model.Dq[i, t] + (self.model.q[i, t] - self.model.q[i, t - 1]))
                self.model.con_dispatch_freq.add(self.model.Dq[i, t] + (self.model.q[i, t] - self.model.q[i, t - 1]) <= self.model.z[i, t] * 1000)
        #
        # # here we try to enforce that if the load is not connected, the associated var cannot be dispatched
        # self.model.status_gen = pm.ConstraintList()
        # for t in self.iter_time:
        #     for i in self.ppc['varcon']:
        #         id = int(i[i.find('_') + 1:])
        #         idx = np.where(id == ppc['gen'][:, 0])[0]
        #         idx = idx[0]  # the original idx was an array, since bus and line idx is unique, we get the entry
        #         bus_id = int(self.ppc["gen"][idx, 0])
        #         bus_name = 'bus_' + str(bus_id)
        #         self.model.status_gen.add(self.model.rho[bus_name, t] >= self.model.ug[i, t])

        ## =================== load status constraints ===================
        self.model.status_load = pm.ConstraintList()
        for t in range(1, self.total_step):
            for i in self.iter_bus:
                self.model.status_load.add(self.model.rho[i, t] >= self.model.rho[i, t - 1])
        # we need to further constraint the initial condition
        for i in self.iter_bus:
            self.model.status_load.add(self.model.rho[i, 0] == Load_Initial[i])

        ##===================== line status constraints=====================
        self.model.status_line = pm.ConstraintList()
        ## static line cannot change
        for t in self.iter_time:
            for i in self.line_static:
                self.model.status_line.add(self.model.ul[i, t] == 1)
        ## Damaged line cannot be on
        for t in self.iter_time:
            for i in self.line_damaged:
                self.model.status_line.add(self.model.ul[i, t] == 0)
        # initial condition is enforced to be equal to Line_Initial
        for i in Line_Initial:
            self.model.status_line.add(self.model.ul[i, 0] == Line_Initial[i])

        ## =========== tie-line for control action ===========
        # closed tieline cannot be opened (not suitiable in learning case)
        for t in range(1, self.total_step):
            for i in self.line_switch:
                self.model.status_line.add(self.model.ul[i, t] >= self.model.ul[i, t-1])
        # in IL/RL setup, we allow one action at a time （must combined with the above constraints)
        for t in range(1, self.total_step):
            self.model.status_line.add(sum(self.model.ul[i, t] for i in self.line_switch) - sum(self.model.ul[i, t - 1] for i in self.line_switch) <= 1)

        # ============= Define objective function =============
        def obj_restoration(model):
            obj = 0 # initialize the objective function
            ## load pickups
            for t in self.iter_time:
                for i in self.iter_bus:
                    id = int(i[i.find('_') + 1:])  # get the matrix index from the component name
                    idx = np.where(id == ppc['bus'][:, 0])[0]
                    idx = idx[0]  # the original idx was an array, since bus and line idx is unique, we get the entry
                    obj = obj - model.rho[i, t] * ppc['bus'][idx, 4]
            # dispatch frequency
            for t in self.iter_time:
                for i in self.ppc['varcon']:
                    obj = obj + 0.01 * self.model.Dq[i, t]
            return obj
        self.model.obj = pm.Objective(rule=obj_restoration)




    def solve_network_restoration(self, Total_Step, Line_Initial, Load_Initial):
        """
        solve a full-step restoration problem
        :param Total_Step: total step of the problem
        :param Line_Initial: initial line status
        :return: restoration plan
        """
        self.model = pm.ConcreteModel()
        self.total_step = Total_Step
        self.iter_time = np.arange(0, Total_Step)
        ppc = self.ppc

        ## ================== network operation variables ====================
        self.model.p = pm.Var(self.iter_gen, self.iter_time, within=pm.Reals)
        self.model.q = pm.Var(self.iter_gen, self.iter_time, within=pm.Reals)
        self.model.P = pm.Var(self.iter_line, self.iter_time, within=pm.Reals)
        self.model.Q = pm.Var(self.iter_line, self.iter_time, within=pm.Reals)
        self.model.V = pm.Var(self.iter_bus, self.iter_time, within=pm.NonNegativeReals)
        self.model.rho = pm.Var(self.iter_bus, self.iter_time, within=pm.Binary)
        self.model.ug = pm.Var(self.iter_gen, self.iter_time, within=pm.Binary)
        self.model.ul = pm.Var(self.iter_line, self.iter_time, within=pm.Binary)
        self.model.beta = pm.Var(self.iter_bus, self.iter_bus, self.iter_time, within=pm.Binary)
        self.model.nb = pm.Var(self.iter_time, within=pm.NonNegativeReals)

        ## =================== distflow constraint: power balance at bus i ===================
        self.model.con_distflow_bus = pm.ConstraintList()  # In pyomo formulation, use ConstraintList() is a generic way to add constraints
        for t in self.iter_time:
            for i in self.iter_bus:
                P_out = 0
                Q_out = 0
                P_in = 0
                Q_in = 0

                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:])
                idx = np.where(id == ppc['bus'][:, 0])[0]
                idx = idx[0]  # the original idx was an array, since bus and line idx is unique, we get the entry

                # get power flow variables flowing out from this bus
                iter_out = np.arange(0, len(self.bus_line[i]["line_from_this_bus"]))

                if iter_out.shape[0] == 0:
                    P_out = P_out
                    Q_out = Q_out
                else:
                    for k in iter_out:
                        P_out = P_out + self.model.P['line_{}'.format(int(self.bus_line[i]["line_from_this_bus"][k])), t]
                        Q_out = Q_out + self.model.Q['line_{}'.format(int(self.bus_line[i]["line_from_this_bus"][k])), t]

                # get power flow variables flowing into from this bus
                iter_in = np.arange(0, len(self.bus_line[i]["line_to_this_bus"]))

                if iter_in.shape[0] == 0:
                    P_in = P_in
                    Q_in = Q_in
                else:
                    for k in iter_in:
                        P_in = P_in + self.model.P['line_{}'.format(int(self.bus_line[i]["line_to_this_bus"][k])), t]
                        Q_in = Q_in + self.model.Q['line_{}'.format(int(self.bus_line[i]["line_to_this_bus"][k])), t]

                # if this is a bus with generation capability and load, the power flow balance is
                if ppc['bus'][idx, 1] == 1:
                    self.model.con_distflow_bus.add(
                        P_in + self.model.p['gen_{}'.format(int(self.bus_gen[i][0])), t] == P_out + ppc['bus'][idx, 4] * self.model.rho[i, t])
                    self.model.con_distflow_bus.add(
                        Q_in + self.model.q['gen_{}'.format(int(self.bus_gen[i][0])), t] == Q_out + ppc['bus'][idx, 5] * self.model.rho[i, t])

                # if this is a bus with only load
                else:
                    self.model.con_distflow_bus.add(P_in == P_out + ppc['bus'][idx, 4] * self.model.rho[i, t])
                    self.model.con_distflow_bus.add(Q_in == Q_out + ppc['bus'][idx, 5] * self.model.rho[i, t])

        ## ===================  distflow constraint: voltage drop along line k ===================
        self.model.con_distflow_line = pm.ConstraintList()
        for t in self.iter_time:
            for i in self.iter_line:
                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:])
                idx = np.where(id == ppc['line'][:, 0])[0]
                idx = idx[0]  # the original idx was an array, since bus and line idx is unique, we get the entry

                # for this line, get the bus index
                f_bus = int(ppc['line'][idx, 1])
                t_bus = int(ppc['line'][idx, 2])

                self.model.con_distflow_line.add(
                    self.model.V['bus_{}'.format(t_bus), t] - self.model.V['bus_{}'.format(f_bus), t]
                    + (ppc['line'][idx, 3] * self.model.P[i, t] + ppc['line'][idx, 4] * self.model.Q[i, t])/self.VS <= (1 - self.model.ul[i, t]) * self.BigM)
                self.model.con_distflow_line.add(
                    self.model.V['bus_{}'.format(t_bus), t] - self.model.V['bus_{}'.format(f_bus), t]
                    + (ppc['line'][idx, 3] * self.model.P[i, t] + ppc['line'][idx, 4] * self.model.Q[i, t])/self.VS >= -(1 - self.model.ul[i, t]) * self.BigM)

        # ======================== Power flow limits =========================
        self.model.con_lim_line = pm.ConstraintList()
        for t in self.iter_time:
            for i in self.iter_line:
                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:])
                idx = np.where(id == ppc['line'][:, 0])[0]
                idx = idx[0]  # the original idx was an array, since bus and line idx is unique, we get the entry
                self.model.con_lim_line.add(self.model.P[i, t] <= self.model.ul[i, t] * ppc['line'][idx, 5])  #ppc['line'][id, 5]
                self.model.con_lim_line.add(self.model.P[i, t] >= -self.model.ul[i, t] * ppc['line'][idx, 5])
                self.model.con_lim_line.add(self.model.Q[i, t] <= self.model.ul[i, t] * ppc['line'][idx, 6]) # ppc['line'][id, 6]
                self.model.con_lim_line.add(self.model.Q[i, t] >= -self.model.ul[i, t] * ppc['line'][idx, 6])

        #  ======================== voltage deviation limits =========================
        self.model.con_lim_voltage = pm.ConstraintList()
        for t in self.iter_time:
            for i in self.iter_bus:
                if i == 'bus_1':
                    self.model.con_lim_voltage.add(self.model.V[i, t] == self.VS)
                else:
                    self.model.con_lim_voltage.add(self.model.V[i, t] >= 1 - self.dV)
                    self.model.con_lim_voltage.add(self.model.V[i, t] <= 1 + self.dV)

        ## ===================== tree topology constraints ==========================
        self.model.con_radiality = pm.ConstraintList()

        # substation bus does not have parent bus
        for t in self.iter_time:
            self.model.con_radiality.add(self.model.beta['bus_2', 'bus_1', t] == 0)  # hard coded

        # each bus will only at least have one parent bus
        for t in self.iter_time:
            for j in self.iter_bus:
                self.model.con_radiality.add(sum(self.model.beta[i, j, t] for i in self.iter_bus) <= 1)

        for t in self.iter_time:
            for i in self.iter_line:
                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:])
                idx = np.where(id == ppc['line'][:, 0])[0]
                idx = idx[0]  # the original idx was an array, since bus and line idx is unique, we get the entry
                # for this line, get the bus index
                f_bus = int(ppc['line'][idx, 1])
                t_bus = int(ppc['line'][idx, 2])
                self.model.con_radiality.add(self.model.beta['bus_{}'.format(f_bus), 'bus_{}'.format(t_bus), t]
                                        + self.model.beta['bus_{}'.format(t_bus), 'bus_{}'.format(f_bus), t] == self.model.ul[i, t])

        # ================== generator dispatch ===============
        # generator dispatch limits
        self.model.con_lim_gen = pm.ConstraintList()
        for t in self.iter_time:
            for i in self.iter_gen:
                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:])
                idx = np.where(id == ppc['gen'][:, 0])[0]
                idx = idx[0]  # the original idx was an array, since bus and line idx is unique, we get the entry
                self.model.con_lim_gen.add(self.model.p[i, t] >= ppc['gen'][idx, 2] * self.model.ug[i, t])
                self.model.con_lim_gen.add(self.model.p[i, t] <= ppc['gen'][idx, 3] * self.model.ug[i, t])
                self.model.con_lim_gen.add(self.model.q[i, t] >= ppc['gen'][idx, 4] * self.model.ug[i, t])
                self.model.con_lim_gen.add(self.model.q[i, t] <= ppc['gen'][idx, 5] * self.model.ug[i, t])

        ## =================== load status constraints ===================
        self.model.status_load = pm.ConstraintList()
        for t in range(1, self.total_step):
            for i in self.iter_bus:
                self.model.status_load.add(self.model.rho[i, t] >= self.model.rho[i, t - 1])
        # we need to further constraint the initial condition
        for i in self.iter_bus:
            self.model.status_load.add(self.model.rho[i, 0] == Load_Initial[i])

        ##===================== line status constraints=====================
        self.model.status_line = pm.ConstraintList()
        ## static line cannot change
        for t in self.iter_time:
            for i in self.line_static:
                self.model.status_line.add(self.model.ul[i, t] == 1)
        ## Damaged line cannot be on
        for t in self.iter_time:
            for i in self.line_damaged:
                self.model.status_line.add(self.model.ul[i, t] == 0)
        # initial condition is enforced to be equal to Line_Initial
        for i in Line_Initial:
            self.model.status_line.add(self.model.ul[i, 0] == Line_Initial[i])

        ## =========== tie-line for control action ===========
        # closed tieline cannot be opened (not suitiable in learning case)
        for t in range(1, self.total_step):
            for i in self.line_switch:
                self.model.status_line.add(self.model.ul[i, t] >= self.model.ul[i, t-1])
        # in IL/RL setup, we allow one action at a time （must combined with the above constraints)
        for t in range(1, self.total_step):
            self.model.status_line.add(sum(self.model.ul[i, t] for i in self.line_switch) - sum(self.model.ul[i, t - 1] for i in self.line_switch) <= 1)

        # ============= Define objective function =============
        def obj_restoration(model):
            obj = 0 # initialize the objective function
            ## load pickups
            for t in self.iter_time:
                for i in self.iter_bus:
                    id = int(i[i.find('_') + 1:])  # get the matrix index from the component name
                    idx = np.where(id == ppc['bus'][:, 0])[0]
                    idx = idx[0]  # the original idx was an array, since bus and line idx is unique, we get the entry
                    obj = obj - model.rho[i, t] * ppc['bus'][idx, 4]
            return obj
        self.model.obj = pm.Objective(rule=obj_restoration)


    def solve_load_pickup(self, line_status, load_status=None):
        """
        Given fixed line status, obtain optimal load status
        -- line status should be a dict with "key" as line name and "int or array" describing status over time interval
        """
        self.model = pm.ConcreteModel()
        Total_Time = 1
        self.total_step = Total_Time
        self.iter_time = np.arange(0, Total_Time)
        ppc = self.ppc

        ## ================== network operation variables ====================
        self.model.p = pm.Var(self.iter_gen, self.iter_time, within=pm.Reals)
        self.model.q = pm.Var(self.iter_gen, self.iter_time, within=pm.Reals)
        self.model.P = pm.Var(self.iter_line, self.iter_time, within=pm.Reals)
        self.model.Q = pm.Var(self.iter_line, self.iter_time, within=pm.Reals)
        self.model.V = pm.Var(self.iter_bus, self.iter_time, within=pm.NonNegativeReals)
        self.model.rho = pm.Var(self.iter_bus, self.iter_time, within=pm.Binary)
        self.model.ug = pm.Var(self.iter_gen, self.iter_time, within=pm.Binary)
        self.model.ul = pm.Var(self.iter_line, self.iter_time, within=pm.Binary)
        self.model.beta = pm.Var(self.iter_bus, self.iter_bus, self.iter_time, within=pm.Binary)
        self.model.nb = pm.Var(self.iter_time, within=pm.NonNegativeReals)

        ## =================== distflow constraint: power balance at bus i ===================
        self.model.con_distflow_bus = pm.ConstraintList()  # In pyomo formulation, use ConstraintList() is a generic way to add constraints
        for t in self.iter_time:
            for i in self.iter_bus:
                P_out = 0
                Q_out = 0
                P_in = 0
                Q_in = 0

                # get the matrix index (idx) from the component name
                id = int(i[i.find('_') + 1:])
                idx = np.where(id == ppc['bus'][:, 0])[0]
                idx = idx[0]  # the original idx was an array, since bus and line idx is unique, we get the entry

                # get power flow variables flowing out from this bus
                iter_out = np.arange(0, len(self.bus_line[i]["line_from_this_bus"]))

                if iter_out.shape[0] == 0:
                    P_out = P_out
                    Q_out = Q_out
                else:
                    for k in iter_out:
                        _id_from = int(self.bus_line[i]["line_from_this_bus"][k])
                        P_out = P_out + self.model.P['line_{}'.format(_id_from), t]
                        Q_out = Q_out + self.model.Q['line_{}'.format(_id_from), t]

                # get power flow variables flowing into from this bus
                iter_in = np.arange(0, len(self.bus_line[i]["line_to_this_bus"]))

                if iter_in.shape[0] == 0:
                    P_in = P_in
                    Q_in = Q_in
                else:
                    for k in iter_in:
                        _id_to = int(self.bus_line[i]["line_to_this_bus"][k])
                        P_in = P_in + self.model.P['line_{}'.format(_id_to), t]
                        Q_in = Q_in + self.model.Q['line_{}'.format(_id_to), t]

                # if this is a bus with generation capability and load, the power flow balance is
                if ppc['bus'][idx, 1] == 1:
                    self.model.con_distflow_bus.add(
                        P_in + self.model.p['gen_{}'.format(self.bus_gen[i][0]), t] == P_out + ppc['bus'][idx, 4] * self.model.rho[i, t])
                    self.model.con_distflow_bus.add(
                        Q_in + self.model.q['gen_{}'.format(self.bus_gen[i][0]), t] == Q_out + ppc['bus'][idx, 5] * self.model.rho[i, t])

                # if this is a bus with only load
                else:
                    self.model.con_distflow_bus.add(P_in == P_out + ppc['bus'][idx, 4] * self.model.rho[i, t])
                    self.model.con_distflow_bus.add(Q_in == Q_out + ppc['bus'][idx, 5] * self.model.rho[i, t])

        ## ===================  distflow constraint: voltage drop along line k ===================
        self.model.con_distflow_line = pm.ConstraintList()
        for t in self.iter_time:
            for i in self.iter_line:
                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:])
                idx = np.where(id == ppc['line'][:, 0])[0]
                idx = idx[0]  # the original idx was an array, since bus and line idx is unique, we get the entry

                # for this line, get the bus index
                f_bus = int(ppc['line'][idx, 1])
                t_bus = int(ppc['line'][idx, 2])

                self.model.con_distflow_line.add(
                    self.model.V['bus_{}'.format(t_bus), t] - self.model.V['bus_{}'.format(f_bus), t]
                    + (ppc['line'][idx, 3] * self.model.P[i, t] + ppc['line'][idx, 4] * self.model.Q[i, t])/self.VS <= (1 - self.model.ul[i, t]) * self.BigM)
                self.model.con_distflow_line.add(
                    self.model.V['bus_{}'.format(t_bus), t] - self.model.V['bus_{}'.format(f_bus), t]
                    + (ppc['line'][idx, 3] * self.model.P[i, t] + ppc['line'][idx, 4] * self.model.Q[i, t])/self.VS >= -(1 - self.model.ul[i, t]) * self.BigM)

        # ======================== Power flow limits =========================
        self.model.con_lim_line = pm.ConstraintList()
        for t in self.iter_time:
            for i in self.iter_line:
                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:])
                idx = np.where(id == ppc['line'][:, 0])[0]
                idx = idx[0]  # the original idx was an array, since bus and line idx is unique, we get the entry

                self.model.con_lim_line.add(self.model.P[i, t] <= self.model.ul[i, t] * ppc['line'][idx, 5])  #ppc['line'][id, 5]
                self.model.con_lim_line.add(self.model.P[i, t] >= -self.model.ul[i, t] * ppc['line'][idx, 5])
                self.model.con_lim_line.add(self.model.Q[i, t] <= self.model.ul[i, t] * ppc['line'][idx, 6]) # ppc['line'][id, 6]
                self.model.con_lim_line.add(self.model.Q[i, t] >= -self.model.ul[i, t] * ppc['line'][idx, 6])

        #  ======================== voltage deviation limits =========================
        self.model.con_lim_voltage = pm.ConstraintList()
        for t in self.iter_time:
            for i in self.iter_bus:
                if i == 'bus_1':
                    self.model.con_lim_voltage.add(self.model.V[i, t] == self.VS)
                else:
                    self.model.con_lim_voltage.add(self.model.V[i, t] >= 1 - self.dV)
                    self.model.con_lim_voltage.add(self.model.V[i, t] <= 1 + self.dV)

        ## ===================== tree topology constraints ==========================
        self.model.con_radiality = pm.ConstraintList()

        # substation bus does not have parent bus
        for t in self.iter_time:
            self.model.con_radiality.add(self.model.beta['bus_2', 'bus_1', t] == 0)  # hard coded

        # each bus will only at least have one parent bus
        for t in self.iter_time:
            for j in self.iter_bus:
                self.model.con_radiality.add(sum(self.model.beta[i, j, t] for i in self.iter_bus) <= 1)

        for t in self.iter_time:
            for i in self.iter_line:
                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:])
                idx = np.where(id == ppc['line'][:, 0])[0]
                idx = idx[0]  # the original idx was an array, since bus and line idx is unique, we get the entry

                # for this line, get the bus index
                f_bus = int(ppc['line'][idx, 1])
                t_bus = int(ppc['line'][idx, 2])
                self.model.con_radiality.add(self.model.beta['bus_{}'.format(f_bus), 'bus_{}'.format(t_bus), t]
                                        + self.model.beta['bus_{}'.format(t_bus), 'bus_{}'.format(f_bus), t] == self.model.ul[i, t])
        # ================== generator dispatch ===============
        # generator dispatch limits
        self.model.con_lim_gen = pm.ConstraintList()
        for t in self.iter_time:
            for i in self.iter_gen:
                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:])
                idx = np.where(id == ppc['line'][:, 0])[0]
                idx = idx[0]  # the original idx was an array, since bus and line idx is unique, we get the entry

                self.model.con_lim_gen.add(self.model.p[i, t] >= ppc['gen'][idx, 2] * self.model.ug[i, t])
                self.model.con_lim_gen.add(self.model.p[i, t] <= ppc['gen'][idx, 3] * self.model.ug[i, t])
                self.model.con_lim_gen.add(self.model.q[i, t] >= ppc['gen'][idx, 4] * self.model.ug[i, t])
                self.model.con_lim_gen.add(self.model.q[i, t] <= ppc['gen'][idx, 5] * self.model.ug[i, t])

        # # ===================== load pickup logic ========================
        if load_status:
            self.model.con_load = pm.ConstraintList()
            for t in self.iter_time:
                for i in self.iter_bus:
                    self.model.con_load.add(self.model.rho[i, t] >= load_status[i])

        ##===================== line status constraints=====================
        self.model.status_line = pm.ConstraintList()
        ## enforce line status to be the same as given values
        for t in self.iter_time:
            for i in self.iter_line:
                self.model.status_line.add(self.model.ul[i, t] == line_status[i])

        # ============= Define objective function =============
        def obj_restoration(model):
            obj = 0 # initialize the objective function
            ## load pickups
            for t in self.iter_time:
                for i in self.iter_bus:
                    id = int(i[i.find('_') + 1:])  # get the matrix index from the component name
                    idx = np.where(id == ppc['bus'][:, 0])[0]
                    idx = idx[0]  # the original idx was an array, since bus and line idx is unique, we get the entry
                    obj = obj - model.rho[i, t] * ppc['bus'][idx, 4]
            return obj
        self.model.obj = pm.Objective(rule=obj_restoration)



    def solve_load_pickup_varcon(self, line_status, varcon, load_status=None):
        """
        Given fixed line status and reactive power dispatch, obtain optimal load status
        -- line status should be a dict with "key" as line name and "int or array" describing status over time interval
        """
        self.model = pm.ConcreteModel()
        Total_Time = 1
        self.total_step = Total_Time
        self.iter_time = np.arange(0, Total_Time)
        ppc = self.ppc

        ## ================== network operation variables ====================
        self.model.p = pm.Var(self.iter_gen, self.iter_time, within=pm.Reals)
        self.model.q = pm.Var(self.iter_gen, self.iter_time, within=pm.Reals)
        self.model.e = pm.Var(self.iter_gen, self.iter_time, within=pm.Reals) # dispatch error to avoid infeasibility
        self.model.ea = pm.Var(self.iter_gen, self.iter_time, within=pm.NonNegativeReals)  # dispatch error to avoid infeasibility
        self.model.eb = pm.Var(self.iter_gen, self.iter_time, within=pm.NonNegativeReals)  # dispatch error to avoid infeasibility
        self.model.ey = pm.Var(self.iter_gen, self.iter_time, within=pm.Binary)  # dispatch error to avoid infeasibility
        self.model.P = pm.Var(self.iter_line, self.iter_time, within=pm.Reals)
        self.model.Q = pm.Var(self.iter_line, self.iter_time, within=pm.Reals)
        self.model.V = pm.Var(self.iter_bus, self.iter_time, within=pm.NonNegativeReals)
        self.model.rho = pm.Var(self.iter_bus, self.iter_time, within=pm.Binary)
        self.model.ug = pm.Var(self.iter_gen, self.iter_time, within=pm.Binary)
        self.model.ul = pm.Var(self.iter_line, self.iter_time, within=pm.Binary)
        self.model.beta = pm.Var(self.iter_bus, self.iter_bus, self.iter_time, within=pm.Binary)
        self.model.nb = pm.Var(self.iter_time, within=pm.NonNegativeReals)

        ## =================== distflow constraint: power balance at bus i ===================
        self.model.con_distflow_bus = pm.ConstraintList()  # In pyomo formulation, use ConstraintList() is a generic way to add constraints
        for t in self.iter_time:
            for i in self.iter_bus:
                P_out = 0
                Q_out = 0
                P_in = 0
                Q_in = 0

                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:])
                idx = np.where(id == ppc['bus'][:, 0])[0]
                idx = idx[0]  # the original idx was an array, since bus and line idx is unique, we get the entry

                # get power flow variables flowing out from this bus
                iter_out = np.arange(0, len(self.bus_line[i]["line_from_this_bus"]))

                if iter_out.shape[0] == 0:
                    P_out = P_out
                    Q_out = Q_out
                else:
                    for k in iter_out:
                        P_out = P_out + self.model.P['line_{}'.format(int(self.bus_line[i]["line_from_this_bus"][k])), t]
                        Q_out = Q_out + self.model.Q['line_{}'.format(int(self.bus_line[i]["line_from_this_bus"][k])), t]

                # get power flow variables flowing into from this bus
                iter_in = np.arange(0, len(self.bus_line[i]["line_to_this_bus"]))

                if iter_in.shape[0] == 0:
                    P_in = P_in
                    Q_in = Q_in
                else:
                    for k in iter_in:
                        P_in = P_in + self.model.P['line_{}'.format(int(self.bus_line[i]["line_to_this_bus"][k])), t]
                        Q_in = Q_in + self.model.Q['line_{}'.format(int(self.bus_line[i]["line_to_this_bus"][k])), t]

                # if this is a bus with generation capability and load, the power flow balance is
                if ppc['bus'][idx, 1] == 1:
                    self.model.con_distflow_bus.add(
                        P_in + self.model.p['gen_{}'.format(int(self.bus_gen[i][0])), t] == P_out + ppc['bus'][idx, 4] * self.model.rho[i, t])
                    self.model.con_distflow_bus.add(
                        Q_in + self.model.q['gen_{}'.format(int(self.bus_gen[i][0])), t] == Q_out + ppc['bus'][idx, 5] * self.model.rho[i, t])

                # if this is a bus with only load
                else:
                    self.model.con_distflow_bus.add(P_in == P_out + ppc['bus'][idx, 4] * self.model.rho[i, t])
                    self.model.con_distflow_bus.add(Q_in == Q_out + ppc['bus'][idx, 5] * self.model.rho[i, t])

        ## ===================  distflow constraint: voltage drop along line k ===================
        self.model.con_distflow_line = pm.ConstraintList()
        for t in self.iter_time:
            for i in self.iter_line:
                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:])
                idx = np.where(id == ppc['line'][:, 0])[0]
                idx = idx[0]  # the original idx was an array, since bus and line idx is unique, we get the entry

                # for this line, get the bus index
                f_bus = int(ppc['line'][idx, 1])
                t_bus = int(ppc['line'][idx, 2])

                self.model.con_distflow_line.add(
                    self.model.V['bus_{}'.format(t_bus), t] - self.model.V['bus_{}'.format(f_bus), t]
                    + (ppc['line'][idx, 3] * self.model.P[i, t] + ppc['line'][idx, 4] * self.model.Q[i, t])/self.VS <= (1 - self.model.ul[i, t]) * self.BigM)
                self.model.con_distflow_line.add(
                    self.model.V['bus_{}'.format(t_bus), t] - self.model.V['bus_{}'.format(f_bus), t]
                    + (ppc['line'][idx, 3] * self.model.P[i, t] + ppc['line'][idx, 4] * self.model.Q[i, t])/self.VS >= -(1 - self.model.ul[i, t]) * self.BigM)

        # ======================== Power flow limits =========================
        self.model.con_lim_line = pm.ConstraintList()
        for t in self.iter_time:
            for i in self.iter_line:
                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:])
                idx = np.where(id == ppc['line'][:, 0])[0]
                idx = idx[0]  # the original idx was an array, since bus and line idx is unique, we get the entry
                self.model.con_lim_line.add(self.model.P[i, t] <= self.model.ul[i, t] * ppc['line'][idx, 5])  #ppc['line'][id, 5]
                self.model.con_lim_line.add(self.model.P[i, t] >= -self.model.ul[i, t] * ppc['line'][idx, 5])
                self.model.con_lim_line.add(self.model.Q[i, t] <= self.model.ul[i, t] * ppc['line'][idx, 6]) # ppc['line'][id, 6]
                self.model.con_lim_line.add(self.model.Q[i, t] >= -self.model.ul[i, t] * ppc['line'][idx, 6])

        #  ======================== voltage deviation limits =========================
        self.model.con_lim_voltage = pm.ConstraintList()
        for t in self.iter_time:
            for i in self.iter_bus:
                if i == 'bus_1':
                    self.model.con_lim_voltage.add(self.model.V[i, t] == self.VS)
                else:
                    self.model.con_lim_voltage.add(self.model.V[i, t] >= 1 - self.dV)
                    self.model.con_lim_voltage.add(self.model.V[i, t] <= 1 + self.dV)

        ## ===================== tree topology constraints ==========================
        self.model.con_radiality = pm.ConstraintList()

        # substation bus does not have parent bus
        for t in self.iter_time:
            self.model.con_radiality.add(self.model.beta['bus_2', 'bus_1', t] == 0)  # hard coded

        # each bus will only at least have one parent bus
        for t in self.iter_time:
            for j in self.iter_bus:
                self.model.con_radiality.add(sum(self.model.beta[i, j, t] for i in self.iter_bus) <= 1)

        for t in self.iter_time:
            for i in self.iter_line:
                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:])
                idx = np.where(id == ppc['line'][:, 0])[0]
                idx = idx[0]  # the original idx was an array, since bus and line idx is unique, we get the entry
                # for this line, get the bus index
                f_bus = int(ppc['line'][idx, 1])
                t_bus = int(ppc['line'][idx, 2])
                self.model.con_radiality.add(self.model.beta['bus_{}'.format(f_bus), 'bus_{}'.format(t_bus), t]
                                        + self.model.beta['bus_{}'.format(t_bus), 'bus_{}'.format(f_bus), t] == self.model.ul[i, t])

        # ================== generator dispatch ===============
        # generator dispatch limits
        self.model.con_lim_gen = pm.ConstraintList()
        for t in self.iter_time:
            for i in self.iter_gen:
                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:])
                idx = np.where(id == ppc['gen'][:, 0])[0]
                idx = idx[0]  # the original idx was an array, since bus and line idx is unique, we get the entry
                self.model.con_lim_gen.add(self.model.p[i, t] >= ppc['gen'][idx, 2] * self.model.ug[i, t])
                self.model.con_lim_gen.add(self.model.p[i, t] <= ppc['gen'][idx, 3] * self.model.ug[i, t])
                self.model.con_lim_gen.add(self.model.q[i, t] >= ppc['gen'][idx, 4] * self.model.ug[i, t])
                self.model.con_lim_gen.add(self.model.q[i, t] <= ppc['gen'][idx, 5] * self.model.ug[i, t])

        # enforce initial conditions of dispatchable generators
        self.model.status_varcon = pm.ConstraintList()
        for t in self.iter_time:
            for i in varcon.keys():
                self.model.status_varcon.add(self.model.e[i, t] == self.model.q[i, t] - varcon[i])

        # supplementary constraint to handle absolute e
        for t in self.iter_time:
            for i in varcon.keys():
                self.model.status_varcon.add(self.model.e[i, t] == self.model.ea[i, t] - self.model.eb[i, t])
                self.model.status_varcon.add(self.model.ea[i, t] <= 0.2 * self.model.ey[i, t])
                self.model.status_varcon.add(self.model.eb[i, t] <= 0.2 * (1 - self.model.ey[i, t]))
        for t in self.iter_time:
            self.model.status_varcon.add(self.model.e['gen_1', t] == 0)
            self.model.status_varcon.add(self.model.ea['gen_1', t] == 0)
            self.model.status_varcon.add(self.model.eb['gen_1', t] == 0)
            self.model.status_varcon.add(self.model.ey['gen_1', t] == 0)

        ## =================== load status constraints ===================
        if load_status:
            self.model.con_load = pm.ConstraintList()
            for t in self.iter_time:
                for i in self.iter_bus:
                    self.model.con_load.add(self.model.rho[i, t] >= load_status[i])

        ##===================== line status constraints=====================
        # enforce line status to be the same as given values
        self.model.status_line = pm.ConstraintList()
        for t in self.iter_time:
            for i in self.iter_line:
                self.model.status_line.add(self.model.ul[i, t] == line_status[i])

        # ============= Define objective function =============
        def obj_restoration(model):
            obj = 0 # initialize the objective function
            ## load pickups
            for t in self.iter_time:
                for i in self.iter_bus:
                    id = int(i[i.find('_') + 1:])  # get the matrix index from the component name
                    idx = np.where(id == ppc['bus'][:, 0])[0]
                    idx = idx[0]  # the original idx was an array, since bus and line idx is unique, we get the entry
                    obj = obj - model.rho[i, t] * ppc['bus'][idx, 4]
            ## minimize the control error between given and feasible var
            for t in self.iter_time:
                for i in varcon.keys():
                    obj = obj + (model.ea[i, t] + model.eb[i, t])
            return obj
        self.model.obj = pm.Objective(rule=obj_restoration)



    def get_solution_2d(self, VariableName, NameKey, ListIndex, SolDict=None):
        """
        get solution and store into a one name key structured dictionary
        :param VariableName: variable name in string format
        :param NameKey: desired key set in list or range format that you would like to retrieve
        :param ListIndex: desired index range in list format that you would like to retrieve
        :param SolDict: dictionary object with plot methods
        :return: SolDict
        """
        if SolDict == None:
            SolDict = SolutionDict()
        else:
            pass

        ## get string type attribute name
        variable = operator.attrgetter(VariableName)(self.model)

        for i in NameKey:
            SolDict[i] = []
            for j in ListIndex:
                SolDict[i].append(variable[i, j].value)

        return SolDict


if __name__=="__main__":

    # import testcase data
    import gym_power_res.envs.data_test_case as case
    ppc = case.case33_tieline()


    # # test solve_network_restoration
    # test_1 = OutageManage()
    # test_1.data_preparation(ppc, ['line_3', 'line_5', 'line_9'])
    # test_1.initialize_problem(5)
    # test_1.solve_network_restoration()
    # opt = pm.SolverFactory("cplex", executable = '/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx/cplex')
    # opt.options['mipgap'] = 0  # if gap=b, then it is (b*100) %
    # results_1 = opt.solve(test_1.model, tee = True)
    # print(results_1['solver'][0]['Termination condition'].key)
    # load_status = test_1.get_solution_2d('rho', test_1.iter_bus, test_1.iter_time)
    # load_status.plot_bin_2d(title_str='load_dg.png', save=False, folder='/Users/whoiszyc/Github/gym-power-res/')
    # line_status = test_1.get_solution_2d('ul', test_1.iter_line, test_1.iter_time)
    # line_status.plot_bin_2d(title_str='line.png', save=False, folder='/Users/whoiszyc/Github/gym-power-res/')