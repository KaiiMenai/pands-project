# analysis.py
# This program will import and analyse the data from the iris dataset. 
# Figures will be output and saved as appropriate.
# author: Kyra Menai Hamilton

# Commencement date: 03/04/2025

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Into terminal: pip install ucimlrepo

# Importing the dataset, fetch the dataset, define the data (as pandas dataframes), print metadata, and print the variable information to check that it worked.
# from ucimlrepo import fetch_ucirepo 

# iris = fetch_ucirepo(id=53) 

# data - extracting x and y (as pandas dataframes) 
# x = iris.data.features 
# y = iris.data.targets 

# metadata - print was to check
# print(iris.metadata) 

# variable information - print was to check
# print(iris.variables) 

# Combine the features and targets into a single DataFrame (df) so it can be exported as a CSV
# iris_df = pd.concat([x, y], axis=1)

# Exporting the DataFrame (df) to a CSV file
# iris_df.to_csv('D:/Data_Analytics/Modules/PandS/pands-project/iris.csv', index=False)
# print("Iris dataset has been successfully exported to a CSV!") # Output - Iris dataset has been successfully exported to a CSV!

# now to retrieve dataset for making plots and analysis
# Import the dataset from the CSV file
iris_df = pd.read_csv('D:/Data_Analytics/Modules/PandS/pands-project/iris.csv')

print(iris_df) # This will print the dataframe into the terminal and also gi ve a brief summary of (150 rows x 5 columns).

# Basic data checks - check for missing values, duplicates, and data types
## Using the 'with' statement to handle file operations


with open("basic_data_explore.txt", "w") as file:
    print("Basic data checks:", file=file)
    print("The shape of the dataset:", file=file)
    print(iris_df.shape, file=file)
    print("The first 5 rows of the dataset:", file=file)
    print(iris_df.head(), file=file) # This will print the first 5 rows of the dataset.
    print("The last 5 rows of the dataset:", file=file)
    print(iris_df.tail(), file=file) # This will print the last 5 rows of the dataset.
    print("Created on: 07/04/2025", file=file)

print("Basic data checks have been written to basic_data_explore.txt")

with open("basic_data_explore.txt", "a") as file:
    print("The number of rows and columns in the dataset:", file=file)
    print(iris_df.info(), file=file) # This will print the number of rows and columns in the dataset.
    print("The number of missing values in the dataset:", file=file)
    print(iris_df.isnull().sum(), file=file) # This will print the number of missing values in the dataset.
    print("The number of duplicate rows in the dataset:", file=file)
    print(iris_df.duplicated().sum(), file=file) # This will print the number of duplicate rows in the dataset.
    print("The data types of each column in the dataset:", file=file)
    print(iris_df.dtypes, file=file) # This will print the data types of each column in the dataset.print("This is the initial content of the file.", file=file)
    print("Created on: 07/04/2025", file=file)
    
print("Basic data checks have been appended to basic_data_explore.txt")

# Need to make sure tha any duplicates are removed and that the data types are correct before conducting any analysis.
# Already checked for missing values and we know there are 0, but there are 3 duplicate rows in the dataset.

data = iris_df.drop_duplicates(subset="class",) # This will remove any duplicate rows in the dataset, based on the class(species) column.

# Summarise each variable in the dataset and check for outliers - export to a single text file.

with open("summary_statistics.txt", "w") as file:  # New file for summary stats
    # Summary statistics for each species
    print("Summary statistics for each species:", file=file)
    
    # Separate the dataset by species
    setosa_stats = iris_df[iris_df['class'] == 'Iris-setosa'].describe()
    versicolor_stats = iris_df[iris_df['class'] == 'Iris-versicolor'].describe()
    virginica_stats = iris_df[iris_df['class'] == 'Iris-virginica'].describe()

    # Display the statistics for each species
    print("Setosa Statistics:", file=file)
    print(setosa_stats, file=file)

    print("\nVersicolor Statistics:", file=file)
    print(versicolor_stats, file=file)

    print("\nVirginica Statistics:", file=file)
    print(virginica_stats, file=file)

print("Summary statistics for each species has been written to summary_statistics.txt")

with open("summary_statistics.txt", "a") as file:  # Append to summary stats file
    # Checking for outliers in the dataset

    # Function to detect outliers using the inter-quartile range method
    def detect_outliers(df, column):
        Q1 = df[column].quantile(0.25)  # First quartile
        Q3 = df[column].quantile(0.75)  # Third quartile
        IQR = Q3 - Q1  # Interquartile range
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers

    # Check for outliers in each numeric column for each species
    numeric_columns = iris_df.select_dtypes(include=[np.number]).columns

    print("\nOutliers detected for each species:", file=file)
    for species in ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']:
        print(f"\nOutliers for {species}:", file=file)
        species_data = iris_df[iris_df['class'] == species]
        for column in numeric_columns:
            outliers = detect_outliers(species_data, column)
            if not outliers.empty:
                print(f"  Column '{column}': {len(outliers)} outliers", file=file)
            else:
                print(f"  Column '{column}': No outliers detected", file=file)

print("Checks for data outliers has been appended to summary_statistics.txt")

# Here good to do boxplots to illustrate the outliers in the dataset.
# Box plots - plot and save box plots for each variable in the dataset and save as a png file.

# Boxplots by species

# Define feature names and their corresponding titles
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
titles = ['Sepal Length by Species', 'Sepal Width by Species', 
          'Petal Length by Species', 'Petal Width by Species']

# Create boxplots for each feature by species
plt.figure(figsize=(12, 8))
for i, feature in enumerate(features):
    ax = plt.subplot(2, 2, i+1)
    sns.boxplot(x='class', y=feature, hue='class', data=iris_df, ax=ax)
    ax.set_title(titles[i])
    ax.set_xlabel('Species')  # Update x-axis label for clarity
    ax.set_ylabel(feature.replace('_', ' ').title())  # Format y-axis label
plt.tight_layout()

# Save the figure as a PNG file
plt.savefig('boxplots_by_species.png')
plt.show()


# Histograms - plot and save histograms for each variable in the dataset as a png file.

# Set up the figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot histogram for Sepal Length
sns.histplot(data=iris_df, x="sepal length", hue="class", kde=False, ax=axes[0, 0], bins=15)
axes[0, 0].set_title("Sepal Length Distribution by Species")
axes[0, 0].set_xlabel("Sepal Length")
axes[0, 0].set_ylabel("Frequency")

# Plot histogram for Sepal Width
sns.histplot(data=iris_df, x="sepal width", hue="class", kde=False, ax=axes[0, 1], bins=15)
axes[0, 1].set_title("Sepal Width Distribution by Species")
axes[0, 1].set_xlabel("Sepal Width")
axes[0, 1].set_ylabel("Frequency")

# Plot histogram for Petal Length
sns.histplot(data=iris_df, x="petal length", hue="class", kde=False, ax=axes[1, 0], bins=15)
axes[1, 0].set_title("Petal Length Distribution by Species")
axes[1, 0].set_xlabel("Petal Length")
axes[1, 0].set_ylabel("Frequency")

# Plot histogram for Petal Width
sns.histplot(data=iris_df, x="petal width", hue="class", kde=False, ax=axes[1, 1], bins=15)
axes[1, 1].set_title("Petal Width Distribution by Species")
axes[1, 1].set_xlabel("Petal Width")
axes[1, 1].set_ylabel("Frequency")

# Adjust layout for better spacing
plt.tight_layout()
# Save the figure for histogram as a PNG file and show
plt.savefig('histograms_by_species.png')
plt.show()

# Scatter plots - plot and save scatter plots for each pair of variables in the dataset as a png file.

# Use a loop to create scatter plots for each pair of variables

# Other analysis types that may be apprropriate - for each ensure that the figure is saved as a png file.
# - Box plots - good for checking for outliers and distributions
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


# FOR SAVING AS A TXT FILE AND APPENDING AS WE GO ON 
## First, create a file with some initial content



## Now, append to the file
#with open("append_example.txt", "a") as file:
#    print("\nThis content is being appended to the file.", file=file)
#    print("Appended on: 2023-09-02", file=file)

#print("Additional content has been appended to append_example.txt")

## Let's check the final content
#print("\nFinal content of the file:")
#with open("append_example.txt", "r") as file:    print(file.read())