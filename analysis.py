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

    # printing output directly to a txt file: https://labex.io/tutorials/python-how-to-redirect-the-print-function-to-a-file-in-python-398057

    # FOR SAVING AS A TXT FILE AND APPENDING AS WE GO ON 
    ## First, create a file with some initial content
    ## Now, append to the file
#with open("append_example.txt", "a") as file:
#    print("\nThis content is being appended to the file.", file=file)
#    print("Appended on: 2023-09-02", file=file)
#print("Additional content has been appended to append_example.txt")
    ## Check the final content
#print("\nFinal content of the file:")
#with open("append_example.txt", "r") as file:    print(file.read())

# Basic data checks - check for missing values, duplicates, and data types
## Using the 'with' statement to handle file operations

with open("basic_data_explore.txt", "w") as file: # The (file=file) argument is important to remember as it makes sure Python knows to write to the file and not the terminal.
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

# Write observations from the basic data checks to a text file.

with open("analysis.txt", "w") as file: # The (file=file) argument is important to remember as it makes sure Python knows to write to the file and not the terminal.
    print("Data Analysis conducted on the Iris Dataset", file=file)
    print("\n", file=file)
    print("The shape of the dataset:", file=file)
    print(iris_df.shape, file=file)
    print("\nThe dataset contains 150 rows of data and 5 columns. The 5 columns are the species of isis flower (here noted as 'class'), and sepal length, sepal width, petal length, and petal width.", file=file)
    print("The first and last five rows of the dataset are printed below, as well as the column names within the dataset.", file=file)
    print("\nThe first 5 rows of the dataset:", file=file)
    print(iris_df.head(), file=file) # first 5 rows.
    print("\nThe last 5 rows of the dataset:", file=file)
    print(iris_df.tail(), file=file) # last 5 rows.
    print("\nThe column names of the dataset:", file=file)
    print(iris_df.columns, file=file) # column names.
    print("\nThese print checks were conducted to ensure that the data was correctly imported and in the correct format.", file=file)

print("Basic data explanation written to analysis.txt")

with open("analysis.txt", "a") as file:
    print("The number of rows and columns in the dataset:", file=file)
    print(iris_df.info(), file=file) # number of rows and columns.
    print("\nThe number of missing values in the dataset:", file=file)
    print(iris_df.isnull().sum(), file=file) # number of missing values.
    print("\nThe number of duplicate rows in the dataset:", file=file)
    print(iris_df.duplicated().sum(), file=file) # number of duplicate rows.
    print("\nThe data types of each column in the dataset:", file=file)
    print(iris_df.dtypes, file=file)
    print("\nMissing values were checked for in the dataset, there were none.", file=file)
    print("If there were missing values, the dataset would need to be cleaned and sorted further before any analysis could be conducted.", file=file)
    print("There were no missing values in this dataset, so further cleaning was unnecessary.", file=file)
    print("\nFrom the information table, it can be seen that where one column has categorical (object) data (class column - also referred to as species for this dataset) \nand the four other columns (sepal length, sepal width, petal length, and petal width) are of the float type (float64) (continuous variables) with non-Null entries. That is, there are no 0 / null~ entries in the dataset.", file=file)
    
print("Basic data checks explanation has been appended to analysis.txt")

# Need to make sure tha any duplicates are removed and that the data types are correct before conducting any analysis.
# Already checked for missing values and we know there are 0, but there are 3 duplicate rows in the dataset.

data = iris_df.drop_duplicates(subset="class",) # This will remove any duplicate rows in the dataset, based on the class(species) column.

# Summarise each variable in the dataset and check for outliers - export to a single text file.

with open("summary_statistics.txt", "w") as file:  # New file for summary stats
    # Summary statistics for each species
    print("Value counts for each of the species:", file=file)
    print(iris_df['class'].value_counts(), file=file)  # Count of each species in the dataset
    print("\nSummary statistics for the whole dataset:", file=file)
    print(iris_df.describe(), file=file)  # Summary statistics for the entire dataset
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

# Write summary stats observations to the analysis.txt file.

with open("analysis.txt", "a") as file:
    print("\nDuplicates were removed from the data using the drop_duplicates function.", file=file)
    print("The code used for this was: \tdata = iris_df.drop_duplicates(subset="'class'".)", file=file)
    print("Value counts for each of the species:", file=file)
    print(iris_df['class'].value_counts(), file=file)
    print("\nIt can be seen that there are 50 samples for each of the three classes (species) of iris, setosa, versicolor, and virginica.", file=file)
    print("\nSummary statistics for the whole dataset:", file=file)
    print(iris_df.describe(), file=file)
    print("\nThe summary statistics for the whole dataset shows that there are 150 samples in the dataset for each of the variables, the value displayed is the non-missing values, and thus it can be verified that the dataset does not have any missing values. ", file=file) # Summary statistics observations taken from my own work for Principles of Data Analytics, with wording changed to match what was required for this project. https://github.com/KaiiMenai/poda-tasks/blob/main/tasks.ipynb
    print("\tPlease Note: Summary statistics observations taken from my own work for the Principles of Data Analytics Module, with wording changed to match what was required for this project. Link: https://github.com/KaiiMenai/poda-tasks/blob/main/tasks.ipynb", file=file)
    print("The Mean, Standard Deviation (std), Minimum (min) and Maximum (max), and the Lower, Median, and Higher Inter-quartile Ranges (25%, 50%, and 75%, respectively) values are displayed for all four features (sepal length, sepal width, petal length, and petal width).", file=file)
    print("The Mean was calculated by dividing the sum of all the values (per feature) by the number of values (150 in this case). The mean for sepal length was 5.84 cm, sepal width was 3.05 cm, petal length was 3.76 cm, and for petal width was 1.20 cm. ", file=file)
    print("\nThe Standard Deviation (std) is a measure of the spread of the data, that is, on average, how much the values deviate from the mean. Sepal length had a mean of 5.84 cm with a std of 0.83, therefore the majority of values deviate by 0.83 cm (+/-) from the mean. \nThe mean for sepal width was 3.05 cm with a std of 0.43 cm, the sepal widths deviated by +/- 0.43 cm from the mean. The feature, sepal width, had less variability (std = 0.43) compared to that of sepal length (std = 0.83). \nFor petal length, the mean was 3.76 cm with a std of 1.76 cm, thus most values for petal length deviated by 1.76 cm (+/-). Petal width had a mean of 1.20 cm with a std of 0.76 cm, the width of petals deviated by +/- 0.76 cm. \nThe measurement with the largest deviation from the mean is the petal length (std = 1.76), this suggests that petal lengths vary more widely across samples compared to the other features.", file=file)
    print("")
    print("\nIn the Summary Statistics for each species, the count shows that there are 50 samples in the dataset for each, the values displayed is the non-missing value, suggesting that there are no missing values present in the dataset.", file=file)
    print("\nSummary statistics for each species:", file=file)
    # Separate the dataset by species
    setosa_stats = iris_df[iris_df['class'] == 'Iris-setosa'].describe()
    versicolor_stats = iris_df[iris_df['class'] == 'Iris-versicolor'].describe()
    virginica_stats = iris_df[iris_df['class'] == 'Iris-virginica'].describe()
    # Display the statistics for each species
    print("\nSetosa Statistics:", file=file)
    print(setosa_stats, file=file)
    print("\nVersicolor Statistics:", file=file)
    print(versicolor_stats, file=file)
    print("\nVirginica Statistics:", file=file)
    print(virginica_stats, file=file)
    
    
print("Summary Stats has been appended to analysis.txt")

# Now to explain what the summary stats are and what they mean - this will be done in the analysis.txt file.

with open("analysis.txt", "a") as file:
    print("\n\tIris Setosa.", file=file)
    print("\nThe mean for sepal length was AAA cm, sepal width was BBB cm, petal length was CCC cm, and for petal width the mean was DDD cm. The mean was calculated by dividing the sum of all the values (per feature) by the number of values (50 in this case, as it is done by species('class')).", file=file)
    print("\nThe standard deviation (std) is a measure of the spread of the data, that is, on average, how much the values deviate from the mean. For sepal length the mean was AAA cm and the std was AAA, therefore most values deviated by AAA cm (+/-) from the mean.",  file=file)
    print("The mean for sepal width was BBB cm and the std was BBB, so most values deviated by +/- BBB cm from the mean.", file=file)
    print("Petal length had a mean of", file=file)

print("Summary Stats explanation has been appended to analysis.txt")

# Here good to do boxplots to illustrate the outliers in the dataset.
# Box plots - plot and save box plots for each variable in the dataset and save as a png file. https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html and https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.savefig.html and https://stackoverflow.com/questions/7906365/matplotlib-savefig-plots-different-from-show 

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

# Describe what the boxplot shows in the analysis.txt file.

with open("analysis.txt", "a") as file:
    print("\n\tBoxplots by Feature.", file=file)
    print("\nBoxplots were plotted for each of the four measured features (sepal length/width and petal length/width), the data in each of these four plots is separated by species. Boxplots make visualising range, potential outliers, the inter-quartile range, and the median of the data more easily.", file=file)
    print("\nThere were nine outliers in total within the dataset between the four sepal/petal features. The Setosa species had three outliers in the data for petal length, and two outliers in the data for petal width. The Virginica species had one outlier for sepal length and two outliers for sepal width. The Versicolor species had the fewest number of outliers with only one outlier throughout the whole dataset, this outlier was for petal length.",  file=file)
    print("On average, Setosa was found to have the shortest sepal length and widest sepal width. Setosa was also found to have the shortest petal length measurements and narrowest petal width. For Versicolor and Virginica, there were some differences visible in the measurements for the four features (sepal length/width, petal length/width), however, there were instances where the feature measurements converged, particularly for sepal length and sepal width. Petal length and petal width displayed differences between species, indicating that these feature measurements may be valuable for classification of Iris species.", file=file)
    print("(https://www.nickmccullum.com/python-visualization/boxplot/ , https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/statistics/data-presentation/box-and-whisker-plots.html).", file=file)

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

with open("analysis.txt", "a") as file:
    print("\n\Histograms by Feature.", file=file)
    print("\nBoxplots were plotted for each of the four measured features (sepal length/width and petal length/width), the data in each of these four plots is separated by species. Boxplots make visualising range, potential outliers, the inter-quartile range, and the median of the data more easily.", file=file)
    print("\nThere were nine outliers in total within the dataset between the four sepal/petal features. The Setosa species had three outliers in the data for petal length, and two outliers in the data for petal width. The Virginica species had one outlier for sepal length and two outliers for sepal width. The Versicolor species had the fewest number of outliers with only one outlier throughout the whole dataset, this outlier was for petal length.",  file=file)
    print("On average, Setosa was found to have the shortest sepal length and widest sepal width. Setosa was also found to have the shortest petal length measurements and narrowest petal width. For Versicolor and Virginica, there were some differences visible in the measurements for the four features (sepal length/width, petal length/width), however, there were instances where the feature measurements converged, particularly for sepal length and sepal width. Petal length and petal width displayed differences between species, indicating that these feature measurements may be valuable for classification of Iris species.", file=file)
    print("(https://www.nickmccullum.com/python-visualization/boxplot/ , https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/statistics/data-presentation/box-and-whisker-plots.html).", file=file)

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

# - Clustering analysis (e. K-means clustering) is used to group similar data points together.
# In this case, K-means clustering is used to group the iris dataset into three clusters (corresponding to the three species).
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


# - Logistic Regression - model taken from my PoDA project, but adapted for this dataset. (https://github.com/KaiiMenai/poda-tasks/blob/main/tasks.ipynb)

# Prepare data for species classification
X_species = iris_df[['sepal length', 'sepal width', 'petal length', 'petal width']] # These are the features that will be used during the classification.
y_species = iris_df['class'] # This is the target variable that is aimed to be predicted.

# Encode species names to numerical values - Encoding species as numerical values (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html )
le = LabelEncoder()
y_species = le.fit_transform(y_species) # setosa = 0, versicolor = 1, virginica = 2

# Split the dataset into an 80 % (training) and 20 % (testing) split.
X_species_train, X_species_test, y_species_train, y_species_test = train_test_split(X_species, y_species, test_size=0.2, random_state=42)

# Create and fit the model
model_species = LogisticRegression(max_iter=200)
model_species.fit(X_species_train, y_species_train)

# Make predictions on the x_species_test set.
y_species_pred = model_species.predict(X_species_test)

# Calculate accuracy - here the predicted species (y_species_pred) with the actual species (y_species_test) are compared, to see how many were predicted correctly.
accuracy = accuracy_score(y_species_test, y_species_pred)

with open("logistic_regression.txt", "w") as file: # The (file=file) argument is important to remember as it makes sure Python knows to write to the file and not the terminal.
    print("Logistic Regression for Species Classification Results:", file=file) # How to do it - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    print(f"Accuracy: {accuracy:.4f}", file=file)
    print("\nClassification Report:", file=file)
    print(classification_report(y_species_test, y_species_pred, target_names=le.classes_), file=file)
    # Example prediction - predict species based on sepal and petal measurements.
    print("\nExample Prediction (measurements in cm).", file=file)
    example_data = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], 
                                columns=['sepal length', 'sepal width', 'petal length', 'petal width'])  # Match feature names to those in the training set
    predicted_species = model_species.predict(example_data)
    print(f"\nPredicted species for {example_data.iloc[0].tolist()}: {le.inverse_transform(predicted_species)[0]}", file=file) # This will print the last 5 rows of the dataset.

print("Logistic regression results have been written to logistic_regression.txt")


# Discuss the pros and cons of each technique and how they may be applied to this dataset. - do this in a text file called 'analysis.txt'.

with open("pros_cons_analysis.txt", "w") as file: 
    print("The pros and cons of each technique and how they may be applied to this dataset.", file=file) 
    print("\nWhy do any of these techniques?", file=file)
    print("\n\t- Basic EDA (Exploratory Data Analysis) to understand the data and its distribution.", file=file)
    print("\n\t- Visualisation to identify patterns and relationships between features (sepal length/width, petal length/width) and their relationships with the target variable (species).", file=file)
    print("\n\t- PCA to reduce dimensionality and visualise the data in a lower-dimensional space.", file=file)
    print("\n\t- Scatter plots to visualise the relationship between features (sepal length/width, petal length/width) and the target variable (species).", file=file)
    print("\n\t- Box plots to identify outliers and understand the distribution of features (sepal length/width, petal length/width) by species.", file=file)
    print("\n\t- Clustering to group similar data points and identify potential clusters in the data.", file=file)
    print("\n\t- Pair plots to visualise the relationships between all pairs of features and the target variable (species).", file=file)
    print("\n\t- K-means clustering to group similar data points and identify potential clusters in the data.", file=file)
    print("\n\t- Classification techniques to predict the species of iris flowers based on their features (sepal length/width, petal length/width).", file=file)
    print("\n\t- Regression techniques to predict continuous values based on features (sepal length/width, petal length/width).", file=file)
    print("\n\t- Linear Regression to model the relationship between features and a continuous target variable (species).", file=file)
    print("\n\t- Logistic Regression to model the relationship between features and a binary or categorical target variable (species).", file=file)
    print("\n", file=file)
    print("\nEach technique used and short notes on them.", file=file)
    print("\n", file=file)
    print("\nExploratory Data Analysis (EDA):", file=file)
    print("\n\tPros: Helps understand the data, identify patterns, and detect any outliers.", file=file)
    print("\n\tCons: May not provide insights into relationships between features and the target variable.", file=file)
    print("\n\tApplication: Can be used to visualise the distribution of features (sepal length/width, petal length/width) and their relationships with the target variable (species).", file=file)
    print("\n", file=file)
    print("\nScatter Plots:", file=file)
    print("\n\tPros: Visualises the relationship between features and the target variable, helps identify patterns and relationships.", file=file)
    print("\n\tCons: May not work well with high-dimensional data, may not capture complex relationships.", file=file)
    print("\n\tApplication: Can be used to visualise the relationship between features (sepal length/width, petal length/width) and the target variable (species).", file=file)
    print("\n", file=file)
    print("\nBox Plots:", file=file)
    print("\n\tPros: Helps identify outliers, visualises the distribution of features by species.", file=file)
    print("\n\tCons: May not work well with high-dimensional data, may not capture complex relationships.", file=file)
    print("\n\tApplication: Can be used to visualise the distribution of features (sepal length/width, petal length/width) by species.", file=file)
    print("\n", file=file)
    print("\nCorrelation Matrix:", file=file)
    print("\n\tPros: Helps identify relationships between features, visualises the correlation between features and the target variable.", file=file)
    print("\n\tCons: May not work well with high-dimensional data, may not capture complex relationships.", file=file)
    print("\n\tApplication: Can be used to visualise the correlation between features (sepal length/width, petal length/width) and the target variable (species).", file=file)
    print("\n", file=file)
    print("\nPrincipal Component Analysis (PCA):", file=file)
    print("\n\tPros: Reduces the complexity of the data, helps visualise high-dimensional data, captures any variance in the data.", file=file)
    print("\n\tCons: May lose some information, sensitive to scaling, may not be interpretable.", file=file)
    print("\n\tApplication: Can be used to visualise the data in a lower-dimensional space, identify clusters, and reduce noise.", file=file)
    print("\n", file=file)
    print("\nK-means Clustering:", file=file)
    print("\n\tPros: Groups similar data points together, helps identify patterns and relationships in the data.", file=file)
    print("\n\tCons: Sensitive to the choice of parameters, may not work well with non-spherical clusters.", file=file)
    print("\n\tApplication: Can be used to identify clusters in the data, which may correspond to different iris flower species.", file=file)
    print("\n", file=file)
    print("\nClassification:", file=file)
    print("\n\tPros: Can be used to predict categorical outcomes (i.e. species), interpretable results.", file=file)
    print("\n\tCons: Sensitive to the choice of parameters, may not work well with imbalanced data (luckily the iris dataset is balanced).", file=file)
    print("\n\tApplication: Can be used to predict the iris flower species based on their features.", file=file)
    print("\n", file=file)
    print("\nRegression:", file=file)
    print("\n", file=file)
    print("\nLinear Regression:", file=file)
    print("\n\tPros: Simple to implement, interpretable coefficients, works well for linear relationships (iris dataset is linear).", file=file)
    print("\n\tCons: Assumes linear relationship between features and target, sensitive to outliers, may not perform well with non-linear data.", file=file)
    print("\n\tApplication: Can be used to predict continuous values (e.g., sepal width / petal width) based on other features (sepal length, petal length).", file=file)
    print("\n", file=file)
    print("\nLogistic Regression:", file=file)
    print("\n\tPros: Simple to implement, interpretable coefficients, works well for binary classification.", file=file)
    print("\n\tCons: Assumes linear relationship between features and log-odds, may not perform well with non-linear data.", file=file)
    print("\n\tApplication: Can be used to predict the probability of a specific species based on features (sepal length/width, petal length/width).", file=file)
    
print("Pros and Cons for each test written to pros_cons_analysis.txt")