from graphics import Graphics
g = Graphics()

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

### COMPARISON KMEANS vs DBSCAN

Recall = [0.478260869565, 0.700483091787]
Precision = [0.53104410441, 0.648550460055]
Specificity = [0.555315136185, 0.620408163265]

fontsz = 20
fontlabel = 20
matplotlib.rcParams['xtick.labelsize'] = fontlabel 
matplotlib.rcParams['ytick.labelsize'] = fontlabel 
matplotlib.rc('axes', edgecolor='black')


N=2
width = 0.2
ind = np.arange(N)
fig, ax = plt.subplots(figsize=(13,6))
rects1 = ax.bar(ind-width, Recall, width, color='r', label='Recall/Sensitivity')
rects2 = ax.bar(ind, Precision, width, color='b', label='Precision')
rects3 = ax.bar(ind+width, Specificity, width, color='black', label='Specificity')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('K-Means', 'DBScan'))
ax.patch.set_facecolor('white')
ax.grid(c='gray')
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
# Put a legend below current axis
legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
          fancybox=True, shadow=True, ncol=5, fontsize=20)
legend.get_frame().set_facecolor('white')