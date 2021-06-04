

import sys
import os
import time
import pandas as pd
import numpy as np
import math
import sympy as sy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyomo.environ import *
from gurobipy import *
import os
import networkx as nx
import matplotlib.pyplot as plt
from collections import OrderedDict


def get_variable_value_gurobi(rho, iter_bus, iter_time):
    RHO = OrderedDict()
    for i in iter_bus:
        RHO[i] = []
        for t in iter_time:
            RHO[i].append(rho[i, t].x)
    return RHO


def plot_binary_evolution(UL, iter_line, str):
    # plot a binary evolution figure
    # U should be a dictionary, where keys will be the iterm
    # In each iterm, a list of 0 and 1 represents the time evolution

    # get size
    y_axis = UL.keys()
    number_y = len(y_axis)
    iter_time = np.arange(0, len(UL[y_axis[0]]))

    # plotting
    plt.figure(figsize=(15, 8))
    plt.xlabel('Time (step)')
    plt.ylabel('Index')
    k=0
    for i in y_axis:
        for t in iter_time:
            if UL[i][t] == False:
                plt.scatter(t, y_axis[k], c='red', s=50, alpha=0.5, edgecolors='none')
            else:
                plt.scatter(t, y_axis[k], c='green', s=50, alpha=0.5, edgecolors='none')
        k = k + 1
    plt.yticks(y_axis, iter_line)
    plt.title(str)
    plt.show()



def crew_dispatch_MIP_pyomo(model, res, vrp, ppc):

    # get number, index and iterator
    iter_time = res['iter_time']

    # vrp
    iter_crew = vrp['iter_crew']
    iter_vertex = vrp['iter_vertex']
    number_vertex = vrp['number_vertex']



    # read data from pyomo optimization model
    for c in iter_crew:
        res[c] = {}

        res[c]['x'] = {}
        res[c]['route'] = [] # empty list
        for i in iter_vertex:
            for j in iter_vertex:
                res[c]['x'][i, j] = value(model.x[c, i, j])
                if value(model.x[c, i, j]) == 1:
                    res[c]['route'].append((i, j))

        res[c]['y'] = {}
        for i in iter_vertex:
            res[c]['y'][i] = value(model.y[c, i])

        res[c]['AT'] = [] # define a list for plotting
        for i in iter_vertex:
            res[c]['AT'].append(value(model.AT[c, i]))

        # compute the available time of component
        res[c]['ava'] = []
        for i in iter_vertex:
            res[c]['ava'].append(value(model.AT[c, i]) + vrp['repair'][c][i])

        res[c]['f'] = {}
        for i in iter_vertex:
            res[c]['f'][i] = []
            for t in iter_time:
                res[c]['f'][i].append(value(model.f[i, t]))

        res[c]['z'] = {}
        for i in iter_vertex:
            res[c]['z'][i] = []
            for t in iter_time:
                res[c]['z'][i].append(value(model.z[i, t]))


    # add route to the graph
    plt.figure(figsize=(7,5))
    for i in res[0]['route']:
        vrp['graph'].add_edge(i[0], i[1])

    nx.draw(vrp['graph'], vrp['fault']['location'], with_labels=True)
    plt.show()


    # # order the vertex by arrival time
    iter_vertex_order = np.array(list(iter_vertex))
    iter_vertex_ordered = iter_vertex_order[list(np.argsort(res[0]['AT']))]
    plt.figure(figsize=(7,5))
    plt.xlabel('Node')
    plt.ylabel('Time')
    plt.bar(iter_vertex_ordered, np.sort(res[0]['AT']), label='Arravel time')
    plt.bar(iter_vertex_ordered, np.sort(res[0]['ava']), label='Available time', alpha=0.5)
    plt.title('Arravel and available time of components (CPLEX)')
    plt.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.5)
    plt.show()


    plt.figure(figsize=(15,5))
    plt.xlabel('Time (step)')
    plt.ylabel('Component name')
    y_axis = np.arange(0, len(iter_vertex))
    k =0
    for i in vrp['ordered_vertex']:
        for t in res['iter_time']:
            if res[0]['z'][i][t] == 0:
                plt.scatter(t, y_axis[k], c='red', s=50, alpha=0.5, edgecolors='none')
            else:
                plt.scatter(t, y_axis[k], c='green', s=50, alpha=0.5, edgecolors='none')
        k = k + 1
    plt.yticks(y_axis, vrp['ordered_vertex'])
    plt.title('Availability of components (CPLEX)')
    # plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
    plt.show()


def network_flow_MIP(model, res, vrp, ppc):
    # get number, index and iterator
    iter_time = res['iter_time']

    # vrp
    iter_crew = vrp['iter_crew']
    iter_vertex = vrp['iter_vertex']
    number_vertex = vrp['number_vertex']

    # network
    number_bus = ppc['number_bus']
    number_line = ppc['number_line']
    number_gen = ppc['number_gen']
    index_bus = ppc['index_bus']
    index_line = ppc['index_line']
    index_gen = ppc['index_gen']
    iter_bus = ppc['iter_bus']
    iter_gen = ppc['iter_gen']
    iter_line = ppc['iter_line']


    res['P'] = {}
    res['Q'] = {}
    res['V'] = {}
    res['ul'] = {}
    res['rho'] = {}
    res['p'] = {}
    res['q'] = {}

    for i in iter_bus:
        res['V'][i] = []
        for t in iter_time:
            res['V'][i].append(value(model.V[i, t]))

    for i in iter_line:
        res['P'][i] = []
        for t in iter_time:
            res['P'][i].append(value(model.P[i, t]))

    for i in iter_line:
        res['Q'][i] = []
        for t in iter_time:
            res['Q'][i].append(value(model.Q[i, t]))

    for i in iter_line:
        res['ul'][i] = []
        for t in iter_time:
            res['ul'][i].append(int(value(model.ul[i, t])))

    for i in iter_bus:
        res['rho'][i] = []
        for t in iter_time:
            res['rho'][i].append(int(value(model.rho[i, t])))

    for i in iter_gen:
        res['p'][i] = []
        res['q'][i] = []
        for t in iter_time:
            res['p'][i].append(value(model.p[i, t]))
            res['q'][i].append(value(model.q[i, t]))

    # Voltage change at each bus w.r.test_1 time
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111, projection='3d')
    for i in iter_bus:
        id = int(i[i.find('_') + 1:])
        xs = iter_time
        ys = res['V'][i]
        ax.bar(xs, ys, zs=id, zdir='y', alpha=0.7)
        # ax.bar(xs, ys, zs=int(ppc["gen"][i, 0]), zdir='y', alpha=0.7, label='Gen {}'.format(int(ppc["gen"][i, 0])))
    # ax.set_xlabel('Hours')
    # ax.set_ylabel('Generator Index')
    # ax.set_zlabel('Power')
    # plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
    # plt.show()
    plt.title('Voltage')

    # Active power change at each bus w.r.test_1 time
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111, projection='3d')
    for i in iter_line:
        id = int(i[i.find('_') + 1:])
        xs = iter_time
        ys = res['P'][i]
        ax.bar(xs, ys, zs=id, zdir='y', alpha=0.7)
        # ax.bar(xs, ys, zs=int(ppc["gen"][i, 0]), zdir='y', alpha=0.7, label='Gen {}'.format(int(ppc["gen"][i, 0])))
    # ax.set_xlabel('Hours')
    # ax.set_ylabel('Generator Index')
    # ax.set_zlabel('Power')
    # plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
    # plt.show()
    plt.title('Active Power')

    plt.figure(figsize=(15, 8))
    plt.xlabel('Time (step)')
    # plt.ylabel('Component name')
    y_axis = np.arange(0, len(iter_line))
    k = 0
    for i in iter_line:
        for t in iter_time:
            if abs(res['ul'][i][t]) <= 0.01:
                plt.scatter(t, y_axis[k], c='red', s=50, alpha=0.5, edgecolors='none')
            else:
                plt.scatter(t, y_axis[k], c='green', s=50, alpha=0.5, edgecolors='none')
        k = k + 1
    plt.yticks(y_axis, iter_line)
    plt.title('Status of line')
    # plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
    plt.show()

    plt.figure(figsize=(15, 8))
    plt.xlabel('Time (step)')
    # plt.ylabel('Component name')
    y_axis = np.arange(0, len(iter_bus))
    k = 0
    for i in iter_bus:
        for t in iter_time:
            if abs(res['rho'][i][t]) <= 0.01:
                plt.scatter(t, y_axis[k], c='red', s=50, alpha=0.5, edgecolors='none')
            else:
                plt.scatter(t, y_axis[k], c='green', s=50, alpha=0.5, edgecolors='none')
        k = k + 1
    plt.yticks(y_axis, iter_bus)
    plt.title('Status of load')
    # plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
    plt.show()

    # plot power
    plt.figure(figsize=(12, 5))
    plt.xlabel('time (hour)')
    plt.ylabel('power (MW)')
    plt.plot(iter_time, res['P']['line_1'])
    plt.title('Power flow at line 1')
    # plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
    plt.show()

    # plot bus voltage
    plt.figure(figsize=(12, 5))
    plt.xlabel('Bus')
    plt.ylabel('Voltage (pu)')
    selected_bus_name = ['bus_24', 'bus_25', 'bus_29', 'bus_30', 'bus_31']
    for i in selected_bus_name:
        plt.plot(iter_time, res['V'][i], label=i)
    plt.title('Voltage of buses ')
    plt.legend(bbox_to_anchor=(1, 1), loc=0, borderaxespad=0.5)
    plt.show()

    return res