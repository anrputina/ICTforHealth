from graphics import Graphics
g = Graphics()

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')


#### RESULTS FROM DETECTIONS ### 
#### ONLY FOR SIMPLER REPRESENTATION AND PLOT IN BAR CHART ###
#Recall = [0.7004, 0.884057971014, 0.855072463768, 0.990338164251]
#Precision = [0.8226, 0.92327176129, 0.929044283052, 1.0]
#Specificity = [0.8489, 0.926530612245, 0.934693877551, 1.0]
#
#fontsz = 20
#fontlabel = 20
#matplotlib.rcParams['xtick.labelsize'] = fontlabel 
#matplotlib.rcParams['ytick.labelsize'] = fontlabel 
#matplotlib.rc('axes', edgecolor='black')
#
#N=4
#width = 0.2
#ind = np.arange(N)
#fig, ax = plt.subplots(figsize=(13,6))
#rects1 = ax.bar(ind-width, Recall, width, color='r', label='Recall/Sensitivity')
#rects2 = ax.bar(ind, Precision, width, color='b', label='Precision')
#rects3 = ax.bar(ind+width, Specificity, width, color='black', label='Specificity')
#ax.set_xticks(ind + width / 2)
#ax.set_xticklabels(('Min Distance', 'Min Distance PCA', 'Bayesian', 'Bayes 2'))
#ax.patch.set_facecolor('white')
#ax.grid(c='gray')
#box = ax.get_position()
#ax.set_position([box.x0, box.y0 + box.height * 0.1,
#                 box.width, box.height * 0.9])
#
## Put a legend below current axis
#legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
#          fancybox=True, shadow=True, ncol=5, fontsize=20)
#legend.get_frame().set_facecolor('white')