# conda env: rsmt
# -*- coding: utf-8 -*-
# @Time        : 2023/4/29 9:10
# @Author      : gwsun
# @Project     : RSMT-main
# @File        : npy2net.py
# @env         : PyCharm
# @Description :
import numpy as np
degree = 40
size = 10000
array = np.load('array_degree{}_num{}.npy'.format(degree, size))
# array = np.load('/home/dbcloud/sgw/RSMT/random1000_5.npy')
with open('net_degree{}_num{}.net'.format(degree, size), 'w') as f:
    f.write('# Routing benchmark generated random\n'
            '# Date    : 2023-04-29\n'
            '# User    : DIAG\n'
            '# Note    : Length unit is dbu\n\n'
            'PARAMETERS\n\n'
            'dbu_per_micron : 2000\n'
            'unit_resistance : 0.0012675 Ohm/dbu\n'
            'unit_capacitance : 8e-20 Farad/dbu\n'
            'driver_resistance : 25.35 OhmNETS\n\n'
            'Nets\n\n')
    for id, net in enumerate(array):
        f.write('Net {} Net_{} {}\n'.format(id, id, degree))
        for i, point in enumerate(net):
            strs = str(i) + ' '
            strs += str(point[0]) + ' ' + str(point[1])
            strs += '\n'
            f.write(strs)
        f.write('\n')
    