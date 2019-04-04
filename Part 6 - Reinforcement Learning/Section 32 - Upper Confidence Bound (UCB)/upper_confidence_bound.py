# Upper Confidence Bound


#here 10 versions of same add r there..
#we show a add if clicked then1 if not 0
#we have 10000 users .. we have dataset which user selected which add

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
import math
N = 10000
d = 10
ads_selected = []       #which add was selected at each round... in the 1st 10round each add will be selected once
                                 #then in the last rounds will will optimize to just one add here 4
numbers_of_selections = [0] * d   #this vector will show how many time each add was selected
sums_of_rewards = [0] * d    #sum of rewards 1 if selected add is clicked or 0 otherwise
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):    #we want each add to run atleast once so this condition
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400               
        if upper_bound > max_upper_bound:   #here we put very high upper bound so that this condition is always true.. 
        #and it is always implemented.. we do this so that each add is shown atleast once
            max_upper_bound = upper_bound
            ad = i                                       #which add has max bound
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()