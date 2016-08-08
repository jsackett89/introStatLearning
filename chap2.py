# -*- coding: utf-8 -*-
"""
Created on Tue Aug 02 15:31:05 2016

@author: sackettj
"""
# Import necessary packages
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import sklearn

# Read in data from web
adv = pd.read_csv("http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv")

# Plot data
plt.figure(1)
plt.scatter(adv['TV'], adv['Sales'])
plt.xlabel("TV Advertising")
plt.ylabel("Total Sales")

plt.figure(2)
plt.scatter(adv['Radio'], adv['Sales'])
plt.xlabel("Radio Advertising")
plt.ylabel("Total Sales")

plt.figure(3)
plt.scatter(adv["Newspaper"], adv["Sales"])
plt.xlabel("Newspaper Advertising")
plt.ylabel("Total Sales")

plt.show()

# Applied exercises

#1 Read in data
college = pd.read_csv("http://www-bcf.usc.edu/~gareth/ISL/College.csv")

# set index
college = college.set_index(['Unnamed: 0'])

#i. describe data
descriptiveStats = college.describe()

#ii. scatterplot matrix
pd.scatter_matrix(college.ix[: , 0:9])

#iii boxplot
college.boxplot(column = 'Outstate', by = 'Private')

#iv Create "Elite" column
college["Elite"] = college['Top10perc'].apply(lambda x: 1 if x > 50 else 0)

# Summary of elite schools
college.boxplot(column = "Outstate", by = "Elite")

#v Histogram
college.hist(column = "Apps")
college.hist(column = "Enroll")
college.hist(column = "Outstate")
college.hist(column = "Expend")
college.hist(column = "Grad.Rate")


# 2. Auto dataset
# Read data
auto = pd.read_table('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data', delim_whitespace = True,
                     header = None, names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin', 'name'],
                     na_values = "?")

# Describe predictors
autoStats = auto.describe()      

# Range of predictors          
autoStats[range(1, auto.shape[1] - 1)].loc['max'] - autoStats[range(1, auto.shape[1] - 1)].loc['min']

# Mean and sd of predictors
autoStats[range(1, auto.shape[1] - 1)].loc[['mean', 'std']]

# Remove 10th - 85th observations
autoSmall = auto.drop(auto.index[list(range(9, 85))])
autoSmallStats = autoSmall.describe()

# Range, mean, sd of remaining obs
autoSmallStats[range(1, auto.shape[1] - 1)].loc['max'] - autoSmallStats[range(1, auto.shape[1] - 1)].loc['min']
autoSmallStats[range(1, auto.shape[1] - 1)].loc[['mean', 'std']]

pd.scatter_matrix(auto[range(auto.shape[1] - 1)])

# 3. Boston housing data
boston = pd.read_table("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", delim_whitespace = True, 
                       header = None, names = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'blk', 'lstat', 'medv'])

# Make/save scatterplot matrix
bostonPlot = pd.scatter_matrix(boston[range(1, boston.shape[1])])
plt.savefig(r'bostonPlot.png')

# Create separate data frame of descriptive stats
bostonStats = boston.describe()

# Look at outlier values for crime, tax, and parent-teacher ratio
boston['crim'].nlargest(25)
boston['tax'].nlargest(25)
boston['ptratio'].nlargest(25)

# Value counts for Charles River proximity
boston['chas'].value_counts()

# Median parent teacher ratio
boston['ptratio'].median()

# Find record that has min median home value
boston.iloc[boston['medv'].idxmin()]

# More than 7/8 rooms per dwelling
boston.ix[boston['rm'] > 7]
boston.ix[boston['rm'] > 8]






