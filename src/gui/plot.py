import math
from pygooglechart import ScatterChart, Axis

__author__ = 'Kohistan'


def plot_histo(values, plot_file):

    """
    "values" should support the iteritems() method and it should return couples of
    integers. they will be plotted using pygooglechart
    """

    # recording extrema for axis creation because pygooglechart doesn't seem to be able to do it
    minx, miny = (0, 0)
    maxx, maxy = (0, 0)
    xlist, ylist = ([0], [0])
    for (x, y) in values.iteritems():
        if x < minx: minx = x
        if y < miny: miny = y
        if maxx < x: maxx = x
        if maxy < y: maxy = y
        xlist.append(x)
        ylist.append(y)
    chart = ScatterChart(400, 300, )
    chart.add_data(xlist)
    chart.add_data(ylist)
    chart.set_axis_range(Axis.BOTTOM, minx, maxx)
    chart.set_axis_range(Axis.LEFT, miny, maxy)
#    x_step = (maxx + 1 - minx) / 10
#    y_step = (maxy + 1 - miny) / 10
#    chart.set_grid(x_step, y_step)
    chart.download(plot_file)