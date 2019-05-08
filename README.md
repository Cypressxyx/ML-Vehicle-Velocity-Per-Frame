# Frame veloicty estimator using a regression classifation.

This program estimates the velocity of a dash cam video at each subsequent frame.

A regression learning model is made to help train the network on a video with 20400 frames.

# Data analysis
Google docs was used to generate the distribution of the velocity data. 
The bins were made in increments of 5. This allows us to have speed groups that range from i < x < i+5.

From the image, we can quickly notice velocity frames with greater than 30 speed were low compared to other speeds. This would go on to affect
the classifcation rate of higher speeding frames.
![data_distribution](data/data_distribution.png)

Oversampling was done to help distribute the data into a more even dataset. This would increase our classification peformance.
The program finds the speed group with the highest number of occurences, and samples all the subsequencet groups appropriately. 

Using pyplot, the distribution of the data was plotted after the oversampling to show that the data was evenly distributed.
![oversampled_data](data/oversampled_data.png)


