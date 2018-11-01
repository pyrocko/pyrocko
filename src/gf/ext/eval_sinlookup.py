#!/usr/bin/env python3
import numpy as num
import matplotlib.pyplot as plt


def load(filename):
    data = num.loadtxt(filename, skiprows=1, delimiter=',').T

    angle = data[0, :]
    valid = data[2, :] != 0.
    error_perc = data[3, :][valid] / data[1, :][valid]

    return angle, data[1, :], data[2, :], error_perc, valid


def plot(filename):
    angle, math, lookup, error_perc, valid = load(filename)

    fig = plt.figure()
    ax = fig.gca()
    ax_error = ax.twinx()

    ax.plot(angle, math, label='Sine Math.h')
    ax.plot(angle, lookup, label='Sine Lookup')
    ax_error.plot(angle[valid], error_perc * 100, label='Error')

    ax_error.set_ylabel('Error Percent')
    ax.set_ylabel('Sine(Angle)')

    ax.grid(alpha=.3)
    ax.legend(loc=1)
    ax.set_xlabel('Angle [deg]')

    plt.show()


plot('sinlookup-8bit.csv')


fns = ['sinlookup.csv', 'sinlookup-6bit.csv', 'sinlookup-8bit.csv']

fig = plt.figure()
ax = fig.gca()

for fn in fns:
    angle, _, _, error, valid = load(fn)
    ax.plot(angle[valid], error * 100., alpha=.75)

ax.set_ylabel('Error Percent')
ax.set_xlabel('Angle [deg]')
ax.grid(alpha=.3)

plt.show()
