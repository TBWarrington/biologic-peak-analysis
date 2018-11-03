#from Bio import SeqIO
import sys
import os
import glob
import csv

import numpy as np
import pandas as pd
import scipy
import cmath
#import peakutils

#path = sys.argv[1]
path = '/mnt/c/Users/Tim/Google Drive/Work/Python/BioLogic'
#path = '/storage/emulated/0/Download/Sync/Work/Python/BioLogic'
output_file = 'test.txt' #sys.argv[2]
output_file_path = os.path.join(path, output_file)

#with open(output_file, 'w') as w_file:

filenames = []

mode = []
o_r = []
error = []
control_changes = []
counter = []
time = []
control_v = []
ewe_v = []
current = []
cycle = []
q = []
p_w = []

#ewe_v is x axis
#current is y axis
try:
    xrange
except NameError:
    xrange = range #python3 xrange = range

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]



for filename in glob.glob(os.path.join(path, '*.mpt')):
    if filename != output_file_path:
        #filenames.append(filename)
#        print(filename)
        try:
            with open(filename, 'rU',  encoding="latin-1") as o_file: #python3
                ewe_v = []
                current = []
                for _ in xrange(57): #.mpt header is 57 lines long (includes legend)
                    next(o_file) # skip headings
                reader=csv.reader(o_file,delimiter='\t')
                for m, o, e, cc, c, t, cv, v, i, cn, q, p in reader:
                    if float(cn) == 3: #and float(o) == 1:
                        ewe_v.append(float(v))
                        current.append(float(i))
    #            print(ewe_v)
    #            print(current) 
        except TypeError:
            with open(filename, 'rU') as o_file: #python2
                ewe_v = []
                current = []
                for _ in xrange(57): #.mpt header is 57 lines long (includes legend)
                    next(o_file) # skip headings
                reader=csv.reader(o_file,delimiter='\t')
                for m, o, e, cc, c, t, cv, v, i, cn, q, p in reader:
                    if float(cn) == 3: #and float(o) == 1:
                        ewe_v.append(float(v))
                        current.append(float(i))
    #            print(ewe_v)
    #            print(current) 

#print (filenames[2])
#print (ewe_v[2])
#print (current[2])
min_x = min(ewe_v)
max_x = max(ewe_v)
min_y = min(current)
max_y = max(current)

#x_y_dict = dict(zip(ewe_v, current))
y_x_dict = dict(zip(current, ewe_v))


from scipy import signal

#x = np.linspace(0,2*np.pi,100)
#y = np.sin(x) + np.random.random(100) * 0.2
yhat = signal.savgol_filter(current, 51, 3) # window size 51, polynomial order 3

# maxima : use builtin function to find (max) peaks
#max_peakind = signal.find_peaks_cwt(current, np.arange(1,1000))
#generate an inverse numpy 1D arr (in order to find minima)
#inv_data = 1./data
# minima : use builtin function fo find (min) peaks (use inversed data)
#min_peakind = signal.find_peaks_cwt(inv_data, np.arange(1,10))

peak_x = []
peak_y = []

yhat_x_dict = dict(zip(yhat, ewe_v))

#for index in max_peakind:
#    peak_x.append(ewe_v[index])
#    peak_y.append(current[index])

peak_y.append(max(yhat))
peak_x.append(yhat_x_dict.get(max(yhat)))

ewe_v_array = np.array(ewe_v)
current_array = np.array(yhat)
ewe_v_diff = np.diff(ewe_v_array)
current_diff = np.diff(current_array)

diff = current_diff/ewe_v_diff
diff_smoothed = signal.savgol_filter(diff, 51, 3) # window size 51, polynomial order 3



#find max
maxima = find_nearest(diff_smoothed[50:500], 0)
maxima_index = diff_smoothed.tolist().index(maxima)
x_curve = ewe_v[maxima_index - 10: maxima_index + 10]
y_curve = yhat[maxima_index - 10: maxima_index + 10]
fit_para = np.polyfit(x_curve, y_curve, 2)
fit_para_fn = np.poly1d(fit_para)

fit_para_der = np.polyder(fit_para_fn)
fit_para_der_fn = np.poly1d(fit_para_fn)

fit_para_der_x = -fit_para_der[1]/fit_para_der[0]
fit_para_der_y = fit_para[0]*(fit_para_der_x**2) + (fit_para[1]*fit_para_der_x) + fit_para[2]
# solve quadratic equation
#ax**2 + bx + c = 0

#d = (fit_para[1]**2) - (4*fit_para[0]*fit_para[2])

# find two solutions
#sol1 = (-fit_para[1]-cmath.sqrt(d))/(2*fit_para[0])
#sol2 = (-fit_para[1]+cmath.sqrt(d))/(2*a)


maxima_y = yhat[maxima_index]
maxima_x = ewe_v[maxima_index]

print(maxima_x)
print(maxima_y)

print(fit_para_der_x)
print(fit_para_der_y)



second_diff = np.diff(diff_smoothed)/ewe_v_diff[0:len(diff)-1]
sec_diff_smoothed = signal.savgol_filter(second_diff, 51, 3) # window size 51, polynomial order 3


x_linear = ewe_v[50:150]
y_linear = current[50:150]

fit = np.polyfit(x_linear, y_linear, 1)
fit_fn = np.poly1d(fit)

'''
from kivy.garden.graph import Graph, MeshLinePlot
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.slider import Slider
from kivy.uix.widget import Widget
from kivy.base import runTouchApp

from math import sin

graph = Graph(xlabel='Ewe/V', ylabel='<I>/mA', x_ticks_minor=5,
x_ticks_major=0.1, y_ticks_major=0.0001,
y_grid_label=True, x_grid_label=True, padding=5,
x_grid=True, y_grid=True, xmin=min_x, xmax=max_x, ymin=min_y, ymax=max_y)
plot = MeshLinePlot(color=[1, 0, 0, 1])
plot.points = [(ewe_v[x], current[x]) for x in range(0, len(ewe_v))]
baseline = MeshLinePlot(color=[0, 1, 0, 1])
baseline.points = [(ewe_v[x], predictions[x]) for x in range(0, len(line_x))]
#baseline.points = [(baseline_values_x[x], baseline_values[x]) for x in range(0, len(baseline_values_x))]
graph.add_plot(plot)
graph.add_plot(baseline)

runTouchApp(graph)

#milk_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/monthly-milk-production-pounds.csv')
#time_series = milk_data['Monthly milk production (pounds per cow)']
#time_series = np.asarray(time_series)

print(baseline_values)
print(line_x)
print(line_y)

print(len(ewe_v))

#print(ewe_v_array)
#print(milk_data)
#print(time_series)

'''
import scipy.interpolate
yToFind = 0
yreduced = diff_smoothed[50:500] - yToFind
freduced = scipy.interpolate.UnivariateSpline(ewe_v_array[50:500], yreduced, s=0)
xx = freduced.roots()[0]
print(xx)

y_interp = scipy.interpolate.UnivariateSpline(ewe_v_array[50:500], current_array[50:500], s=0)
yy = y_interp(xx)
print(yy)



import numpy as np
from matplotlib import pyplot as plt

x = ewe_v
y = current

#print(max_peakind)

plt.plot(x, y, x, fit_fn(x))#, x[200:500])#, fit_para_fn(x[200:500]), x[200:500])#, fit_para_der_fn(x[200:500]))#, x[406], y[406])
plt.plot(x,yhat, color='red')
plt.scatter(peak_x,peak_y)
plt.scatter(maxima_x, maxima_y, color='red')
plt.scatter(xx, yy, color='green')
#plt.scatter(fit_para_der_x, fit_para_der_y, color='yellow')
#plt.plot(ewe_v_array[0:len(diff_smoothed)],diff_smoothed, color='green')
#plt.plot(ewe_v_array[0:len(sec_diff_smoothed)],sec_diff_smoothed, color='yellow')
#plt.plot(x, y_interp(x), color='yellow')
plt.xlabel('Ewe/V')
plt.ylabel('<I>/mA')
plt.show()
'''
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
import matplotlib.pyplot as plt

x = ewe_v
y = current

plt.plot(x, y, x, fit_fn(x))
plt.plot(x,yhat, color='red')
plt.scatter(peak_x,peak_y)
plt.scatter(maxima_x, maxima_y, color='red')
plt.plot(ewe_v_array[0:len(diff_smoothed)],diff_smoothed, color='green')
plt.xlabel('Ewe/V')
plt.ylabel('<I>/mA')

class MyApp(App):

    def build(self):
        box = BoxLayout()
        box.add_widget(FigureCanvasKivyAgg(plt.gcf()))
        return box

MyApp().run()
'''

