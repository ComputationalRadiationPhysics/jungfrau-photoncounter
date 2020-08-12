#/bin/python

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir

core_configs = [int(f[5:].split("_")[0]) for f in listdir(".") if f[0:5] == "cores" and "_0_10_" in f]

means = []
stddevs = []

for conf in core_configs:
	raw_result = np.loadtxt("cores"+str(conf)+"_0_10_1_1_0_0_.. .. .. data_pool px_101016 allpede_250us_1243__B_000000.dat_.. .. .. data_pool px_101016 gainMaps_M022.bin_.. .. .. data_pool px_101016 Insu_6_tr_1_45d_250us__B_000000.dat.txt")
	
	means += [np.mean(raw_result[3:-1])]
	stddevs += [np.std(raw_result[3:-1])]
df = pd.DataFrame(data={(str(core_configs[i]), means[i]) for i in range(len(core_configs))}, columns=['Core Count', 'Runtime [in s]'])
df['Runtime [in s]'] /= 1000000000.

df.sort_values('Core Count').plot.scatter(x='Core Count', y='Runtime [in s]', title="Runtime with Different Core Configurations\n(100 Frame Packages with 100 Frames each; icc -O3 -march=native; Xeon Gold 6148)\n")
plt.savefig('runtimes.png')

speedup_df = pd.DataFrame(data={(str(core_configs[i]), means[i]) for i in range(len(core_configs))}, columns=['Core Count', 'Speedup']).sort_values('Core Count', ascending=True)
speedup_df['Speedup'] = float(speedup_df.loc[speedup_df['Core Count'] == '1']['Speedup']) / speedup_df['Speedup']

speedup_df.plot.scatter(x='Core Count', y='Speedup', title="Speedup with Different Core Configurations\n(100 Frame Packages with 100 Frames each; icc -O3 -march=native; Xeon Gold 6148)\n")
plt.savefig('speedup.png')
