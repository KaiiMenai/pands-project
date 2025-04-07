# analysis.py
# This program will import and analyse the data from the iris dataset. 
# Figures will be output and saved as appropriate.
# author: Kyra Menai Hamilton

# Commencement date: 03/04/2025

# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Into terminal: pip install ucimlrepo

# Importing the dataset, fetch the dataset, define the data (as pandas dataframes), print metadata, and print the variable information to check that it worked.
from ucimlrepo import fetch_ucirepo 

iris = fetch_ucirepo(id=53) 

# data - extracting x and y (as pandas dataframes) 
x = iris.data.features 
y = iris.data.targets 

# metadata - print was to check
# print(iris.metadata) 

# variable information - print was to check
# print(iris.variables) 

# Combine the features and targets into a single DataFrame (df) so it can be exported as a CSV
iris_df = pd.concat([x, y], axis=1)

# Exporting the DataFrame (df) to a CSV file
iris_df.to_csv('D:/Data_Analytics/Modules/PandS/pands-project/iris.csv', index=False)
print("Iris dataset has been successfully exported to a CSV!") # Output - Iris dataset has been successfully exported to a CSV!

# now to retrieve dataset for making plots and analysis
# Import the dataset from the CSV file
iris_df = pd.read_csv('D:/Data_Analytics/Modules/PandS/pands-project/iris.csv')

print(iris_df) # This will print the dataframe into the terminal and also gi ve a brief summary of (150 rows x 5 columns).

# Basic data checks - check for missing values, duplicates, and data types
## Using the 'with' statement to handle file operations
with open("output_with.txt", "w") as output_file:
    print("This line is written using the 'with' statement.", file=output_file)
    print("The file will be automatically closed after this block.", file=output_file)
    print("Numbers and other data types:", 100, 3.14159, True, file=output_file)

print("Output has been written to output_with.txt")

# Summarise each variable in the dataset and check for outliers - export to a single text file.


# Histograms - plot and save histograms for each variable in the dataset as a png file.
# Use seaborn for better aesthetics


# Scatter plots - plot and save scatter plots for each pair of variables in the dataset as a png file.
# Use seaborn for better aesthetics
# Use a loop to create scatter plots for each pair of variables

# Other analysis types that may be apprropriate - for each ensure that the figure is saved as a png file.
# - Box plots
# - Pair plots
# - Correlation matrix
# - Heatmaps
# - PCA (Principal Component Analysis)
# - Clustering analysis (e.g., K-means clustering)


# This is a placeholder for the other analysis types that may be appropriate. - DELETE THIS LINE


# This dataset has previously been used for machine learning and classification tasks, so it may be useful to explore some of those techniques as well.
# - Linear Regression
# - Logistic Regression
# Discuss the pros and cons of each technique and how they may be applied to this dataset.
# printing output directly to a txt file: https://labex.io/tutorials/python-how-to-redirect-the-print-function-to-a-file-in-python-398057