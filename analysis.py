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
    print("The column names of the dataset:", file=file)
    print(iris_df.columns, file=file) # This will print the column names of the dataset.
    
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
    ax.set_xlabel("Species")  # Update x-axis label for clarity
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
axes[0, 0].set_xlabel("Sepal Length (cm)")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].legend(title="Species", labels=iris_df['class'].unique(), loc='upper right')

# Plot histogram for Sepal Width
sns.histplot(data=iris_df, x="sepal width", hue="class", kde=False, ax=axes[0, 1], bins=15)
axes[0, 1].set_title("Sepal Width Distribution by Species")
axes[0, 1].set_xlabel("Sepal Width (cm)")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].legend(title="Species", labels=iris_df['class'].unique(), loc='upper right')

# Plot histogram for Petal Length
sns.histplot(data=iris_df, x="petal length", hue="class", kde=False, ax=axes[1, 0], bins=15)
axes[1, 0].set_title("Petal Length Distribution by Species")
axes[1, 0].set_xlabel("Petal Length (cm)")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].legend(title="Species", labels=iris_df['class'].unique(), loc='upper right')

# Plot histogram for Petal Width
sns.histplot(data=iris_df, x="petal width", hue="class", kde=False, ax=axes[1, 1], bins=15)
axes[1, 1].set_title("Petal Width Distribution by Species")
axes[1, 1].set_xlabel("Petal Width (cm)")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].legend(title="Species", labels=iris_df['class'].unique(), loc='upper right')

# Adjust layout for better spacing
plt.tight_layout()
# Save the figure for histogram as a PNG file and show
plt.savefig('histograms_by_species.png')
plt.show()

# Scatter plots - plot and save scatter plots for each pair of variables in the dataset as a png file.

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Scatter plot for sepal length vs width
sns.scatterplot(ax=axes[0], data=iris_df, x='sepal length', y='sepal width', hue='class', s=100)
axes[0].set_title('Sepal Length vs Sepal Width by Species')
axes[0].set_xlabel('Sepal Length (cm)')
axes[0].set_ylabel('Sepal Width (cm)')
axes[0].legend(title="Species")
axes[0].grid(True)

# Scatter plot for petal length vs width
sns.scatterplot(ax=axes[1], data=iris_df, x='petal length', y='petal width', hue='class', s=100)
axes[1].set_title('Petal Length vs Petal Width by Species')
axes[1].set_xlabel('Petal Length (cm)')
axes[1].set_ylabel('Petal Width (cm)')
axes[1].legend(title="Species")
axes[1].grid(True)

plt.tight_layout()
plt.savefig('scatterplot_by_species.png')
plt.show()

# Other analysis types that may be appropriate - for each ensure that the figure is saved as a png file.
# - Pair plots

pairplot = sns.pairplot(iris_df, hue='class', height=2.5)
# give the plot a title
plt.suptitle("Pairwise Feature Relationship", y=1.02)
pairplot._legend.set_title('Species')
# Save the figure for pairplot as a PNG file and show
plt.savefig('pairplot_by_species.png')
plt.show()

# - Correlation matrix
corr_matrix = iris_df.iloc[:, :4].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.savefig('correlation_matrix_by_feature.png')
plt.show()

# - PCA (Principal Component Analysis)

X = iris_df.iloc[:, :4] # The first four columns are the features (sepal length, sepal width, petal length, petal width)
# The last column is the target variable (species).
y = iris_df['class']
# However, we need to standardise the data before performing PCA.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled[:5])  # Print first 5 rows of scaled data
# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
# Visualize PCA results
plt.figure(figsize=(8, 6))
sns.scatterplot(x=principal_components[:, 0], y=principal_components[:, 1], 
                hue=iris_df['class'], s=100) # Scatter plot of the first two principal components
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})') # PC1 is the first principal component, which is the petal length and width
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})') # PC2 is the second principal component, which is the sepal length and width
plt.legend(title="Species")
plt.title("Principal Component Analysis of the Iris Dataset")
plt.savefig('pca_by_species.png')
plt.show()

# - Clustering analysis (e. K-means clustering)
# clustering analysis is a technique used to group similar data points together.
# In this case, we will use K-means clustering to group the iris dataset into three clusters (corresponding to the three species).
from sklearn.cluster import KMeans 

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # 3 clusters for the 3 species
iris_df['cluster'] = kmeans.fit_predict(iris_df.iloc[:, :4])  # Fit on the first 4 columns (features)

# Map cluster labels to species for consistent coloring
cluster_to_species = {
    0: 'Iris-setosa',
    1: 'Iris-versicolor',
    2: 'Iris-virginica'
}
iris_df['cluster_species'] = iris_df['cluster'].map(cluster_to_species)

# Plot the K-means clustering results
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=iris_df['sepal length'], 
    y=iris_df['sepal width'], 
    hue=iris_df['class'], 
    s=100
)
plt.title("K-means Clustering of Iris Dataset")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend(title="Species")
plt.savefig('kmeans_clustering_by_species.png')
plt.show()


# This dataset has previously been used for machine learning and classification tasks, so it may be useful to explore some of those techniques as well.
# - Linear Regression

# Sepal Length vs Sepal Width
plt.figure(figsize=(10, 6))
X_sepal = iris_df[['sepal length']]
y_sepal = iris_df['sepal width']
model_sepal = LinearRegression()
model_sepal.fit(X_sepal, y_sepal)
y_sepal_pred = model_sepal.predict(X_sepal)
r2_sepal = r2_score(y_sepal, y_sepal_pred)

sns.scatterplot(data=iris_df, x='sepal length', y='sepal width', hue='class', s=100)
sns.regplot(data=iris_df, x='sepal length', y='sepal width', scatter=False, color='red')
plt.title('Sepal Length vs Sepal Width by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend(title='Species')
plt.grid(True)
plt.text(0.05, 0.95, f'R² = {r2_sepal:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
plt.tight_layout()
plt.savefig('lrm_sepal_length_vs_width.png')  # Save the plot as a PNG file
plt.show()

# Petal Length vs Petal Width
plt.figure(figsize=(10, 6))
X_petal = iris_df[['petal length']]
y_petal = iris_df['petal width']
model_petal = LinearRegression()
model_petal.fit(X_petal, y_petal)
y_petal_pred = model_petal.predict(X_petal)
r2_petal = r2_score(y_petal, y_petal_pred)

sns.scatterplot(data=iris_df, x='petal length', y='petal width', hue='class', s=100)
sns.regplot(data=iris_df, x='petal length', y='petal width', scatter=False, color='red')
plt.title('Petal Length vs Petal Width by Species')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend(title='Species')
plt.grid(True)
plt.text(0.05, 0.95, f'R² = {r2_petal:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
plt.tight_layout()
plt.savefig('lrm_petal_length_vs_width.png')  # Save the plot as a PNG file
plt.show()

# Linear Regression and R² values for Sepal Length vs Sepal Width and Petal Length vs Petal Width
# The R² value indicates how well the model explains the variance in the data.
# A higher R² value indicates a better fit.

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Sepal Length vs Sepal Width
sns.scatterplot(ax=axes[0], data=iris_df, x='sepal length', y='sepal width', hue='class', s=100)
axes[0].set_title('Sepal Length vs Sepal Width by Species')
axes[0].set_xlabel('Sepal Length (cm)')
axes[0].set_ylabel('Sepal Width (cm)')
axes[0].grid(True)

# Fit and plot regression lines for each species
for species in iris_df['class'].unique():
    subset = iris_df[iris_df['class'] == species]
    X = subset[['sepal length']]
    y = subset['sepal width']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    sns.lineplot(ax=axes[0], x=subset['sepal length'], y=y_pred, label=f'{species} (R²={r2:.2f})')
    
# Add legend title for the first subplot
axes[0].legend(title='Species and Regression')

# Petal Length vs Petal Width
sns.scatterplot(ax=axes[1], data=iris_df, x='petal length', y='petal width', hue='class', s=100)
axes[1].set_title('Petal Length vs Petal Width by Species')
axes[1].set_xlabel('Petal Length (cm)')
axes[1].set_ylabel('Petal Width (cm)')
axes[1].grid(True)

# Fit and plot regression lines for each species
for species in iris_df['class'].unique():
    subset = iris_df[iris_df['class'] == species]
    X = subset[['petal length']]
    y = subset['petal width']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    sns.lineplot(ax=axes[1], x=subset['petal length'], y=y_pred, label=f'{species} (R²={r2:.2f})')

# Add legend title for the second subplot
axes[1].legend(title='Species and Regression')

plt.tight_layout()
plt.savefig('linear_regression_and_r2_by_species.png')  # Save the plot as a PNG file
plt.show()


# - Logistic Regression



# Discuss the pros and cons of each technique and how they may be applied to this dataset. - do this in a text file called 'analysis.txt'.
# The pros and cons of each technique and how they may be applied to this dataset.
# Why do any of these techniques?
# - Basic EDA (Exploratory Data Analysis) to understand the data and its distribution.
# - Visualisation to identify patterns and relationships between features and the target variable.
# - PCA to reduce dimensionality and visualize the data in a lower-dimensional space.
# - Clustering to group similar data points and identify potential clusters in the data.
# - Linear Regression to model the relationship between features and a continuous target variable.
# - Logistic Regression to model the relationship between features and a binary or categorical target variable.
# - K-means clustering to group similar data points and identify potential clusters in the data.
# - Classification techniques to predict the species of iris flowers based on their features.
# - Regression techniques to predict continuous values based on features.
# Exploratory Data Analysis (EDA):
# Pros: Helps understand the data, identify patterns, and detect any outliers.
# Cons: May not provide insights into relationships between features and the target variable.
# Application: Can be used to visualise the distribution of features (sepal length/width, petal length/width) and their relationships with the target variable (species).
# It can help identify potential features for modeling and inform feature engineering decisions.
# Principal Component Analysis (PCA):
# Pros: Reduces the complexity of the data, helps visualise high-dimensional data, captures any variance in the data.
# Cons: May lose some information, sensitive to scaling, may not be interpretable.
# Application: Can be used to visualize the data in a lower-dimensional space, identify clusters, and reduce noise.
# It can help improve the performance of machine learning algorithms by reducing complexity and removing noise.

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