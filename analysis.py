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
import textwrap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Into terminal: pip install ucimlrepo

# Importing the dataset, fetch the dataset,
# define the data (as pandas dataframes),
# print metadata,
# and print the variable information to check that it worked.
# from ucimlrepo import fetch_ucirepo

# iris = fetch_ucirepo(id=53)

# data - extracting x and y (as pandas dataframes)
# x = iris.data.features
# y = iris.data.targets

# metadata - print was to check
# print(iris.metadata)

# variable information - print was to check
# print(iris.variables)

# Combine the features and targets into a single df to export as CSV
# iris_df = pd.concat([x, y], axis=1)

# Exporting the DataFrame (df) to a CSV file
# iris_df.to_csv
#   ('D:/Data_Analytics/Modules/PandS/pands-project/iris.csv', index=False)
# print("Iris dataset has been successfully exported to a CSV!")
# Output - Iris dataset has been successfully exported to a CSV!

# Now to retrieve dataset for making plots and analysis
# Import the dataset from the CSV file

iris_df = pd.read_csv('D:/Data_Analytics/Modules/PandS/pands-project/iris.csv')
print(iris_df)
# This will print df into terminal & also give brief summary (150 rs x 5 cols).

# printing output directly to a txt file:
# https://labex.io/tutorials/python-how-to-redirect-the-print-function-to-a-file-in-python-398057
#    FOR SAVING AS A TXT FILE AND APPENDING AS WE GO ON
#    First, create a file with some initial content
#    Now, append to the file
# with open("append_example.txt", "a") as file:
#     print("\nThis content is being appended to the file.", file=file)
#     print("Appended on: 2023-09-02", file=file)
# print("Additional content has been appended to append_example.txt")
#    Check the final content
# print("\nFinal content of the file:")
# with open("append_example.txt", "r") as file:    print(file.read())

# The (file=file) argument is important as it makes sure Python knows
# to write the file and not in the terminal.

with open("analysis.md", "w") as file:
    print("# Data Analysis conducted on the Iris Dataset", file=file)
    print("\n## Summary", file=file)
    summary_text = (
    "The analysis used as a source for this document (analysis.py) was conducted using Python and the Pandas, Numpy, Matplotlib, Seaborn, and Scikit-learn libraries. The dataset was imported from the UCI Machine Learning Repository and is a well-known dataset for classification tasks. "
    "The dataset contained 150 samples of iris flowers, with 5 columns: sepal length, sepal width, petal length, petal width, and species (class). The dataset was used to conduct exploratory data analysis (EDA) and visualisation, as well as some machine learning tasks. Histogram, boxplot, scatterplot, PCA, pairplot, K-means clustering, correlation matrix, and linear regression analysis were conducted on the dataset, and the results of each were saved as a PNG file."
    )
    print("", file=file)
    print(textwrap.fill(summary_text, width=210), file=file)
    print("\n*Please Note: Some observations taken from my own work for the Principles of Data Analytics Module, with wording changed to match what was required for this project. Link: https://github.com/KaiiMenai/poda-tasks/blob/main/tasks.ipynb*", file=file)

print("Summary of file made in the analysis.md")

with open("analysis.md", "a") as file:
    print("\n## Background", file=file)
    background1_text = (
    "Originally sourced by Anderson (1935), the Iris dataset has been used numerous times, with several different iterations available online. Some of these sources contain differing (and often noted as incorrect) data points, as noted in the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/53/iris). The dataset contained 150 samples of Iris flower, each with five noted variables, four numeric (sepal and petal feature measurements), and one categorical (the three species), giving a total of 750 sample points throughout the entire dataset."
    " Fisher (1936) made the Iris dataset well known through his paper titled 'The Use of Multiple Measurements in Taxonomic Problems.' In the paper, the multivariate Iris dataset was used as an example of linear discriminant analysis. That is, a statistical method used to find a linear combination of features that can either characterise or separate two or more classes of objects or events (https://en.wikipedia.org/wiki/Linear_discriminant_analysis; https://www.ibm.com/think/topics/linear-discriminant-analysis)."
    )
    background2_text = (
    "Anderson (1935) originally collected the iris samples to study species variation and hybidisation. Anderson (1935) used the dataset to quantify the morphological differences and variation between Iris species, focussing on the evolution of the Versicolor species, and how it may have come about as a hybrid of the Setosa and Virginica Iris species. An interesting point about the dataset is that two of the three species, Iris Versicolor and Iris Virginica, were collected from the same pasture, on the same day, and measured using the same equipment. This is noteworthy for analysis, as Virginica and Versicolor often appear to converge and are not as easily separable as the Setosa species (histograms, scatter plots, etc.)."
    " The Iris dataset has been extensively used as a training dataset, a learning dataset, and for developing machine learning techniques. The scikit-learn library in Python uses the Iris dataset for demonstration purposes and explains how algorithms can learn from data samples (features) to predict class labels (https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html; https://archive.ics.uci.edu/dataset/53/iris)."
    )
    background3_text = (
    "The Iris dataset also highlights the distinction between supervised and unsupervised learning. Supervised learning uses labeled data to train models for classification or regression, while unsupervised learning explores patterns or clusters in unlabeled data (https://www.ibm.com/think/topics/linear-discriminant-analysis). The iris dataset gives an example of supervised learning problems, particularly multi-class classification, where the goal is to predict an iris flower's species based on its measurements."
    " Although the dataset only includes flower measurements (sepal length/width and petal length/width) and not measurements for the entire plant, this focus reflects the expert judgment of Fisher and Anderson, who selected petal and sepal dimensions as meaningful discriminative features. This in turn highlights the importance of domain expertise in data preparation and model design, suggesting that more efficient analysis and models are built when field experts are involved (Domingos, 2012; Kuhn, and Johnson, 2013)."
    " Despite its age and simplicity, the Iris dataset remains a central teaching tool for introducing classification problems in machine learning."
    )
    print("", file=file)
    print(textwrap.fill(background1_text, width=210), file=file)
    print("", file=file)
    print(textwrap.fill(background2_text, width=210), file=file)
    print("", file=file)
    print(textwrap.fill(background3_text, width=210), file=file)

print("Background Intro appended to analysis.md")

# Basic data checks - check for missing values, duplicates, and data types
# Using the 'with' statement to handle file operations
# Write observations from the basic data checks to a text file.

with open("analysis.md", "a") as file: # The (file=file) argument is important to remember as it makes sure Python knows to write to the file and not the terminal.
    print("\n## Exploratory Data Analysis", file=file)
    print("\nThe shape of the dataset:", file=file)
    print(iris_df.shape, file=file)
    print("\nThe dataset contains 150 rows of data and 5 columns. The 5 columns are the species of isis flower (here noted as 'class'), and sepal length, sepal width, petal length, and petal width.", file=file)
    print("", file=file)
    print("The first and last five rows of the dataset are printed below, as well as the column names within the dataset.", file=file)
    print("\nThe first 5 rows of the dataset:", file=file)
    print(iris_df.head().to_markdown(), file=file) # first 5 rows.
    print("\nThe last 5 rows of the dataset:", file=file)
    print(iris_df.tail().to_markdown(), file=file) # last 5 rows.
    print("\nThe column names of the dataset:", file=file)
    print(iris_df.columns, file=file) # column names.
    print("\nThese print checks were conducted to ensure that the data was correctly imported and in the correct format.", file=file)

print("Basic data explanation written to analysis.md")

with open("analysis.md", "a") as file:
    print("The number of rows and columns in the dataset:", file=file)
    print(iris_df.info(), file=file) # number of rows and columns.
    print("\nThe number of missing values in the dataset:", file=file)
    print(iris_df.isnull().sum().to_markdown(), file=file) # number of missing values.
    print("\nThe number of duplicate rows in the dataset:", file=file)
    print(iris_df.duplicated().sum(), file=file) # number of duplicate rows.
    print("\nThe data types of each column in the dataset:", file=file)
    print(iris_df.dtypes.to_markdown(), file=file)
    eda_text = (
    "Missing values were checked for in the dataset, there were none."
    " If there were missing values, the dataset would need to be cleaned and sorted further before any analysis could be conducted."
    " There were no missing values in this dataset, so further cleaning was unnecessary."
    " From the information table, it can be seen that where one column has categorical (object) data (class column - also referred to as species for this dataset) \nand the four other columns (sepal length, sepal width, petal length, and petal width) are of the float type (float64) (continuous variables) with non-Null entries. That is, there are no 0 / null~ entries in the dataset."
    )
    print("", file=file)
    print(textwrap.fill(eda_text, width=210), file=file)

print("Basic data checks explanation has been appended to analysis.md")

# Need to make sure tha any duplicates are removed and that the data types are correct before conducting any analysis.
# Already checked for missing values and we know there are 0, but there are 3 duplicate rows in the dataset.

data = iris_df.drop_duplicates(subset="class",) # This will remove any duplicate rows in the dataset, based on the class(species) column.

# Summarise each variable in the dataset and check for outliers - export to the md file.

# Write summary stats observations to the analysis.md file.

with open("analysis.md", "a") as file:
    print("\nDuplicates were removed from the data using the drop_duplicates function.", file=file)
    print("The code used for this was: ```data = iris_df.drop_duplicates(subset="'class'".)```", file=file)
    print("", file=file)
    print("Value counts for each of the species:", file=file)
    print(iris_df['class'].value_counts(), file=file)
    print("\nIt can be seen that there are 50 samples for each of the three classes (species) of iris flower: Setosa, Versicolor, and Virginica.", file=file)
    print("\nSummary statistics for the whole dataset:", file=file)
    print(iris_df.describe().to_markdown(), file=file)
    iris_summary_text = (
    "The summary statistics for the whole dataset shows that there are 150 samples in the dataset for each of the variables, the value displayed is the non-missing values, and thus it can be verified that the dataset does not have any missing values."
    " The Mean, Standard Deviation (std), Minimum (min) and Maximum (max), and the Lower, Median, and Higher Inter-quartile Ranges (25%, 50%, and 75%, respectively) values are displayed for all four features (sepal length, sepal width, petal length, and petal width)."
    " The Mean was calculated by dividing the sum of all the values (per feature) by the number of values (150 in this case). The mean for sepal length was 5.84 cm, sepal width was 3.05 cm, petal length was 3.76 cm, and for petal width was 1.20 cm."
    " The Standard Deviation (std) is a measure of the spread of the data, that is, on average, how much the values deviate from the mean. Sepal length had a mean of 5.84 cm with a std of 0.83, therefore the majority of values deviate by 0.83 cm (+/-) from the mean. \nThe mean for sepal width was 3.05 cm with a std of 0.43 cm, the sepal widths deviated by +/- 0.43 cm from the mean. The feature, sepal width, had less variability (std = 0.43) compared to that of sepal length (std = 0.83). \nFor petal length, the mean was 3.76 cm with a std of 1.76 cm, thus most values for petal length deviated by 1.76 cm (+/-). Petal width had a mean of 1.20 cm with a std of 0.76 cm, the width of petals deviated by +/- 0.76 cm. \nThe measurement with the largest deviation from the mean is the petal length (std = 1.76), this suggests that petal lengths vary more widely across samples compared to the other features."
    )
    print("", file=file)
    print(textwrap.fill(iris_summary_text, width=210), file=file)
# Checking for outliers in the dataset
# Function to detect outliers using the inter-quartile range method
    def detect_outliers(df, column):
        Q1 = df[column].quantile(0.25)# First quartile
        Q3 = df[column].quantile(0.75)# Third quartile
        IQR = Q3 - Q1# Interquartile range
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
    print("\nIn the Summary Statistics for each species, the count shows that there are 50 samples in the dataset for each, the values displayed is the non-missing value, suggesting that there are no missing values present in the dataset.", file=file)

print("Summary Stats has been appended to analysis.md")
print("Checks for data outliers has been appended to analysis.md")

# Now to explain what the summary stats are and what they mean - this will be done in the analysis.txt file.

with open("analysis.md", "a") as file:
    print("\n## Individual Species Observations", file=file)
# Separate the dataset by species
    setosa_stats = iris_df[iris_df['class'] == 'Iris-setosa'].describe()
    versicolor_stats = iris_df[iris_df['class'] == 'Iris-versicolor'].describe()
    virginica_stats = iris_df[iris_df['class'] == 'Iris-virginica'].describe()
# Display the statistics for each species
    print("\n### Iris Setosa", file=file)
    print("", file=file)
    print("Setosa Statistics:", file=file)
    print(setosa_stats.to_markdown(), file=file)
    setosa_stats_summary_text = (
    "The mean for sepal length was AAA cm, sepal width was BBB cm, petal length was CCC cm, and for petal width the mean was DDD cm. The mean was calculated by dividing the sum of all the values (per feature) by the number of values (50 in this case, as it is done by species('class'))."
    " The standard deviation (std) is a measure of the spread of the data, that is, on average, how much the values deviate from the mean. For sepal length the mean was AAA cm and the std was AAA, therefore most values deviated by AAA cm (+/-) from the mean."
    " The mean for sepal width was BBB cm and the std was BBB, so most values deviated by +/- BBB cm from the mean."
    " Petal length had a mean of XXXXXXXX"
    )
    print("", file=file)
    print(textwrap.fill(setosa_stats_summary_text, width=210), file=file)
    print("\n### Iris Versicolor", file=file)
    print("", file=file)
    print("Versicolor Statistics:", file=file)
    print(versicolor_stats.to_markdown(), file=file)
    versicolor_stats_summary_text = (
    "The mean for sepal length was AAA cm, sepal width was BBB cm, petal length was CCC cm, and for petal width the mean was DDD cm. The mean was calculated by dividing the sum of all the values (per feature) by the number of values (50 in this case, as it is done by species('class'))."
    " The standard deviation (std) is a measure of the spread of the data, that is, on average, how much the values deviate from the mean. For sepal length the mean was AAA cm and the std was AAA, therefore most values deviated by AAA cm (+/-) from the mean."
    " The mean for sepal width was BBB cm and the std was BBB, so most values deviated by +/- BBB cm from the mean."
    " Petal length had a mean of XXXXXXXX"
    )
    print("", file=file)
    print(textwrap.fill(versicolor_stats_summary_text, width=210), file=file)
    print("\n### Iris Virginica", file=file)
    print("", file=file)
    print("\nVirginica Statistics:", file=file)
    print(virginica_stats.to_markdown(), file=file)    
    virginica_stats_summary_text = (
    "The mean for sepal length was AAA cm, sepal width was BBB cm, petal length was CCC cm, and for petal width the mean was DDD cm. The mean was calculated by dividing the sum of all the values (per feature) by the number of values (50 in this case, as it is done by species('class'))."
    " The standard deviation (std) is a measure of the spread of the data, that is, on average, how much the values deviate from the mean. For sepal length the mean was AAA cm and the std was AAA, therefore most values deviated by AAA cm (+/-) from the mean."
    " The mean for sepal width was BBB cm and the std was BBB, so most values deviated by +/- BBB cm from the mean."
    " Petal length had a mean of XXXXXXXX"
    )
    print("", file=file)
    print(textwrap.fill(virginica_stats_summary_text, width=210), file=file)

print("Summary Stats for each species has been appended to analysis.md")

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

with open("analysis.md", "a") as file:
    print("\n## Boxplots by Feature", file=file)
    print("", file=file)
    print("![Boxplots](boxplots_by_species.png)", file=file)
    boxplot_text = (
    "Boxplots were plotted for each of the four measured features (sepal length/width and petal length/width), the data in each of these four plots is separated by species. Boxplots make visualising range, potential outliers, the inter-quartile range, and the median of the data more easily."
    " There were nine outliers in total within the dataset between the four sepal/petal features. The Setosa species had three outliers in the data for petal length, and two outliers in the data for petal width. The Virginica species had one outlier for sepal length and two outliers for sepal width. The Versicolor species had the fewest number of outliers with only one outlier throughout the whole dataset, this outlier was for petal length."
    " On average, Setosa was found to have the shortest sepal length and widest sepal width. Setosa was also found to have the shortest petal length measurements and narrowest petal width. For Versicolor and Virginica, there were some differences visible in the measurements for the four features (sepal length/width, petal length/width), however, there were instances where the feature measurements converged, particularly for sepal length and sepal width. Petal length and petal width displayed differences between species, indicating that these feature measurements may be valuable for classification of Iris species."
    " (https://www.nickmccullum.com/python-visualization/boxplot/ , https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/statistics/data-presentation/box-and-whisker-plots.html)."
    )
    print("", file=file)
    print(textwrap.fill(boxplot_text, width=210), file=file)
    
print("Boxplot observations appended to analysis.md")

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

with open("analysis.md", "a") as file:
    print("\n## Histograms by Feature", file=file)
    print("", file=file)
    print("![Histograms](histograms_by_species.png)", file=file)
    histogram_text = (
    "The histogram plots are all colour coded by species; blue for Setosa, orange for Versicolor, and green for Virginica."
    " From the histogram plot for sepal length by species, the Setosa species showed a normal distribution, with the majority of sepals being approximately 5.0 cm in length. The Versicolor species has a broad range of sepal lengths, with the most sepals being approximately 5.5 cm in length. The species with the largest range in length of sepals and longest average sepal length is the Virginica species."
    " For sepal width, it can be seen that overall, the data shows a normal distribution. Contrary to what was observed for sepal length, the narrowest sepal width is the Versicolor species, with the Virginica species being in the middle of the range. The Setosa species had the greatest width and the broadest range in values for sepal width."
    " Similar to what was observed for sepal length, for petal length Setosa was the species with the shortest average length and the smallest range in measurements. An average petal length of approximately 4.5 cm was observed for the Versicolor species and demonstrated a normal distribution. The Virginica species had, on average, the longest petal lengths, similar to what was observed for sepal lengths."
    " Setosa species had the narrowest petal width on average. The species with the mid-width measurement was the Versicolor species, with values between 1.0 cm and 2.0 cm. The widest petal widths were observed in the Virginica species."
    " It was observed that the sepal width and petal width for the Setosa species were contrary to one another. For the petal measurements of length and width, the Setosa species was the shortest and narrowest and the values for this species also separated away from the other two species."
    )
    print("", file=file)
    print(textwrap.fill(histogram_text, width=210), file=file)

print("Histogram observations appended to analysis.md")

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

with open("analysis.md", "a") as file:
    print("\n## Scatterplots", file=file)
    print("", file=file)
    print("![Scatterplots](scatterplot_by_species.png)", file=file)
    print("", file=file)
    scatter_plot_text = (
    ""
    )
    print("", file=file)
    print(textwrap.fill(scatter_plot_text, width=210), file=file)

print("Scatter plot observations appended to analysis.md")

# Other analysis types that may be appropriate - for each ensure that the figure is saved as a png file.
# - Pair plots

pairplot = sns.pairplot(iris_df, hue='class', height=2.5)
# give the plot a title
plt.suptitle("Pairwise Feature Relationship", y=1.02)
pairplot._legend.set_title('Species')
# Save the figure for pairplot as a PNG file and show
plt.savefig('pairplot_by_species.png')
plt.show()

with open("analysis.md", "a") as file:
    print("\n## Pairplot", file=file)
    pairplot1_text = (
    "A pairplot was used to visualise comparisons between pairs of features, sepal length vs, sepal width, sepal length vs petal length, sepal length vs petal width, petal length vs petal width etc. for the three species of iris flower (Setosa in blue, Versicolor in orange, Verginica in green). Pairwise analysis outputs multiple sub-plots that are plotted in a matrix format; row name gives the x axis, column name gives the y axis, and univariate distributions (histograms) are plotted on the diagonal from top left to bottom right for each feature (https://medium.com/analytics-vidhya/pairplot-visualization-16325cd725e6; https://www.analyticsvidhya.com/blog/2024/02/pair-plots-in-machine-learning/; https://seaborn.pydata.org/generated/seaborn.pairplot.html; https://builtin.com/articles/seaborn-pairplot)."
    )
    print(textwrap.fill(pairplot1_text, width=210), file=file)
    print("", file=file)
    print("![Pairplot](pairplot_by_species.png)", file=file)
    print("", file=file)
    pairplot2_text = (
    "The pairplot displays a good overall view of the relationships between the feature variables, and due to each species having a different colour, it is easy to identify whether or not the species separate out from one another. For different pairs of features following the pairwise pairplot comparison(s) there are different levels of overlap and / or separation seen. For sepal width vs sepal length, a high level of overlap is seen between the Versicolor and Virginica species, Setosa separates out from the other two species and shows some variation in values. Petal length vs sepal length, showed the Setosa species demonstrated complete separation and clustering, some separation was seen between the Versicolor and Virginica species. The distribution of data for petal width vs sepal length showed separation and clustering of the Setosa species with Versicolor and Virginica only displaying slight overlap. Petal length vs sepal width, demonstrates that all species show some variation in their values, the Setosa species clusters separately from the other two, and there was only a small level of overlap in the Versicolor and Virginica species. Similarly to that of petal width vs sepal length, petal width vs sepal width demonstrated clustering of the Setosa species with Versicolor and Virginica only overlapping a very small amount. Petal width vs petal length demonstrated the most distinct separation between all species, there was a small level of overlap between Versicolor and Virginica, however, these features offered the best clustering of the data (https://seaborn.pydata.org/generated/seaborn.pairplot.html; https://toxigon.com/seaborn-pairplot-comprehensive-guide?utm_source=chatgpt.com; https://toxigon.com/seaborn-pairplot-guide?utm_source=chatgpt.com; https://builtin.com/articles/seaborn-pairplot)."
    )
    print("", file=file)
    print(textwrap.fill(pairplot2_text, width=210), file=file)

print("Pairplot observations appended to analysis.md")

# - Correlation matrix
corr_matrix = iris_df.iloc[:, :4].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.savefig('correlation_matrix_by_feature.png')
plt.show()

with open("analysis.md", "a") as file:
    print("\n## Correlation Matrix", file=file)
    print("", file=file)
    print("![Correlation_Matrix](correlation_matrix_by_feature.png)", file=file)
    print("", file=file)
    corr_matrix_text = (
    ""
    )
    print("", file=file)
    print(textwrap.fill(corr_matrix_text, width=210), file=file)

print("Correlation Matrix observations appended to analysis.md")

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

with open("analysis.md", "a") as file:
    print("\n## Principal Component Analysis (PCA)", file=file)
    pca_text = (
    "A way to conduct relationship investigations is through Principal Component Analysis (PCA) - I did this for my PhD research and found it was a great way to clearly look at multiple data aspects at once (https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html)."
    "An important note of PCA is that the data needs to be standardised for it. When standardising data, it's important that it is scaled correctly, otherwise the results will appear skewed and purely incorrect.The analysis can then be run again. The standardised PCA analysis can then be viewed in a plot."
    "The Principal Component Analysis (PCA) transforms the original variables (sepal length, sepal width, petal length, petal width) into a new set of variables that are linear combinations of the original data, known as principal components (Jolliffe and Cadima, 2016). The first principal component (PC1) captures the maximum variance within the data, whilst the second principal component (PC2) captures the remaining variance that is perpendicular (orthogonal) to PC1. Any components following the first and second capture the remaining variance, again perpendicular to all previous components (Jolliffe and Cadima, 2016). Insights following the PCA show that PC1 strongly correlated with petal features, suggesting that petal length and petal width are responsible for the majority, 72.8 %, of the variance within the data. PC2 captured the variance for the sepal length and width, these were responsible for 23 % of the variance in the data. These results demonstrate that the first two components explain 95.8 % of the variance within the dataset."
    "With so much variance seen in PC1 for the petal features, it could indicate that these are good determining factors for use in species identification."
    )
    print("", file=file)
    print(textwrap.fill(pca_text, width=210), file=file)

print("Principal Component Analysis (PCA) observations appended to analysis.md")

# - Clustering analysis (e. K-means clustering) is used to group similar data points together.
# In this case, K-means clustering is used to group the iris dataset into three clusters (corresponding to the three species).

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

with open("analysis.md", "a") as file:
    print("\n## K-means Clustering", file=file)
    k_means_text = (
    ""
    )
    print("", file=file)
    print(textwrap.fill(k_means_text, width=210), file=file)

print("K-means Clustering observations appended to analysis.md")

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

with open("analysis.md", "a") as file:
    print("\n## Linear Regression and R^2 by Species", file=file)
    print("", file=file)
    print("![Linear Regression & R2 by species](linear_regression_and_r2_by_species.png)", file=file)
    print("", file=file)
    linear_regression_and_r2_by_species1_text = (
    "For both plots the species are colour coded (Setosa is blue, Versicolor is orange, Verginica is green)."
    )
    linear_regression_and_r2_by_species2_text = (
    "The left plot displays the Linear Regression for the Sepal features (Length vs Width). Setosa, shown in blue, had an R<sup>2</sup> value of 0.56. Versicolor (orange), showed an R<sup>2</sup> value of 0.28. Virginica, in green, had an R<sup>2</sup> value of 0.21. Of the three species, Setosa is the species where sepal features, length and width, may influence the variance in one another. However, due to the overlap between Versicolor (R<sup>2</sup> = 0.28) and Virginica (R<sup>2</sup> = 0.21), other factors may be responsible for the variance observed between the Sepal features."
    )
    linear_regression_and_r2_by_species3_text = (
    "In the right plot (Petal Length vs Petal Width), the three species Setosa (blue), Versicolor (orange), and Virginica (green) Linear Regression for the Petal Features is plotted. The three species have clearly separated from one another. However, although on a dataset wide basis Petal Features have influence over one another (R<sup>2</sup> = 0.93), at species level, the results are quite different. Out of the three species, the only one to display influence on variance for petal features over one another was the Virginica species (R<sup>2</sup> = 0.62), with the Setosa and Virginica species showing that other factors may be responsible for the variance between these features, with R<sup>2</sup> values of 0.09 and 0.10, respectively. (James *et al*., 2013, https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html, https://medium.com/%40brandon93.w/regression-model-evaluation-metrics-r-squared-adjusted-r-squared-mse-rmse-and-mae-24dcc0e4cbd3, https://www.geeksforgeeks.org/ml-mathematical-explanation-of-rmse-and-r-squared-error/)"
    )
    print("", file=file)
    print(textwrap.fill(linear_regression_and_r2_by_species1_text, width=210), file=file)
    print("", file=file)
    print(textwrap.fill(linear_regression_and_r2_by_species2_text, width=210), file=file)
    print("", file=file)
    print(textwrap.fill(linear_regression_and_r2_by_species3_text, width=210), file=file)

# INSERT LINEAR OBSERVATION HERE

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

with open("analysis.md", "a") as file:
    print("\n## Logistic Regression Model", file=file)
    log_reason_text = (
    "Although a linear regression model has previously been used to analyse the data, it is possible to predict a species based on the measurements taken from a flower through using a logistic regression model."
    " Using a Logistic Regression Model on the iris dataset is appropriate due to the relatively small size of the dataset (Log Regression Models work well with small, linearly separable datasets)."
    " The model is good for multi-class classification, as seen in the dataset with the 'species' classifications. When looking at the results of the logistic regression model, there are some things that should be taken into consideration."
    " The iris dataset is relatively small, and when it is split on a species level, it has even fewer values. Although Logistic Regression Models work well on small datasets, splitting the dataset into multiple species creates a condensed dataset \nand may, not be representative of the wider population. (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html; https://www.ibm.com/think/topics/logistic-regression)"
    )
    print("", file=file)
    print(textwrap.fill(log_reason_text, width=210), file=file)

print("Logistic regression reasoning has been appended to analysis.md")

with open("analysis.md", "a") as file: # The (file=file) argument is important to remember as it makes sure Python knows to write to the file and not the terminal.
    print("\nLogistic Regression for Species Classification Results:", file=file) # How to do it - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    print(f"Accuracy: {accuracy:.4f}", file=file)
    print("\nClassification Report:", file=file)
    print(classification_report(y_species_test, y_species_pred, target_names=le.classes_), file=file)
    # Example prediction - predict species based on sepal and petal measurements.
    print("Example Prediction (measurements in cm).", file=file)
    example_data = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], 
                                columns=['sepal length', 'sepal width', 'petal length', 'petal width'])  # Match feature names to those in the training set
    predicted_species = model_species.predict(example_data)
    print(f"\nPredicted species for {example_data.iloc[0].tolist()}: {le.inverse_transform(predicted_species)[0]}", file=file) # This will print the last 5 rows of the dataset.

print("Logistic regression results have been written to analysis.md")

with open("analysis.md", "a") as file:
    log_regression1_text = (
    "For the iris dataset, the Logistic Regression Model achieves high accuracy (~ 97 %), this is due to the simplicity and linear separability of the iris dataset. Accuracy for the Logistic Regression Model is calculated as the ratio of correct predictions to total predictions. The columns for precision and recall measure class specific performance in the model, and the f1-score column gives a balances metric for imbalanced classes (not a problem here as the iris dataset is balanced). In the support column, it can be seen that the Verginica species has a value of 9, demonstrating a slightly lower recall, this is likely due to the overlap with Versicolor. As seen throughout previous testing on the dataset, the majority of the confusion in the dataset occurs between the Versicolor and Virginica species."
    )
    log_regression2_text = (
    "Initially upon looking at the Classification Report, it could be assumed that the results do not highlight anything. This report provides detailed insights into the performance of the model for each species, showing areas where the model performs well or struggles. However, due to the size of the dataset (150 samples), and then the test set being even smaller (30 samples), it makes it easier for the model to achieve perfect accuracy, this model produced an accuracy of 1.0 (perfect accuracy). A linearly separable dataset is one that shows clear distinctions between classes, for the iris dataset that class difference is seen most clearly in the petal length/width. As the iris dataset is balanced, with an equal number of samples for each class (50), the risk of bias in the model is reduced. In order to improve accuracy and reliability, the model should be rerun a number of times using different splits of the data, this is called cross-validation."
    )
    print("", file=file)
    print(textwrap.fill(log_regression1_text, width=210), file=file)
    print("", file=file)
    print(textwrap.fill(log_regression2_text, width=210), file=file)

print("Logistic Regression observations appended to analysis.md")

# Confusion matrix
# Generate the confusion matrix
cm = confusion_matrix(y_species_test, y_species_pred)

# Plot the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix for Species Classification")
plt.xlabel('Predicted Species')
plt.ylabel('Actual Species')
plt.tight_layout()
plt.savefig('confusion_matrix_species.png')  # Save the plot as a PNG file
plt.show()

with open("analysis.md", "a") as file:
    print("\n## Confusion Matrix", file=file)
    con_matrix_text = (
    "A confusion matrix was plotted to visualise the results. The confusion matrix is a performance evaluation tool for classification models. It provides a summary of the prediction results by comparing the actual values (rows) against the predicted values (columns)."
    " The confusion matrix helps with understanding how well the logistic regression model classifies the different species of iris and whether there are any species that are more prone to misclassification (https://www.analyticsvidhya.com/blog/2020/04/confusion-matrix-machine-learning/)."
    " To interpret the matrix, the structure and values within the matrix are important to understand what it shows. The rows within the matrix represent the Actual Classes (actual species of iris), whilst the columns represent the Predicted Classes (predicted species from the model output)."
    " The matrix contains a number of values, on the diagonal line (from top left to bottom right) the values denote the Correct Predictions, where the actual and predicted classes (species) match. All other values from the diagonal are those denoting Misclassifications, where the actual and predicted species differ."
    " Although the Logistic Regression Model gives a value for Accuracy (1.000) for species classification, the classification report and confusion matrix aid in giving a clearer picture of the data and the accuracy of predictions made with the model."
    )
    print("", file=file)
    print(textwrap.fill(con_matrix_text, width=210), file=file)

print("Confusion Matrix observations appended to analysis.md")

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

## Conclusion - write a conclusion to the analysis.txt file.

with open("analysis.md", "a") as file:
    print("\n## Conclusion", file=file)
    conclusion_text = (
    "The analysis of the Iris dataset has provided valuable insights into the relationships between features (sepal length/width, petal length/width) and the target variable (species)."
    " The dataset was found to be balanced, with 50 samples for each of the three species (Setosa, Versicolor, Virginica). The summary statistics showed that there were no missing values in the dataset, and the features had different means and standard deviations. The box plots and histograms provided visualisations of the distribution of features by species, and the scatter plots showed the relationships between features and the target variable."
    " For a number of plots (histograms, scatter plots, linear regression, box plots) the Setosa species clearly clusters separately to the Versicolor and Virginica species, for both sepal and petal features. For petal features, all species separated from one another (with some minor overlap visible between the Versicolor and Virginica species), indicating that petal features were more distinguishable between species, and thus would be more useful in classification."
    " The PCA and K-means clustering techniques were used to reduce the dimensionality of the data and identify clusters in the data, respectively. Following the principal component analysis (PCA) the observations about petal features being more distinguishable between species, commpared to sepal features was solidified as it was found that PC1 (first principal component) was responsible for 72.8 % of the variability seen within the data, the PC1 was referring to the petal length and width features."
    " The feature correlation matrix heat map also displayed the difference between species based onn petal features, where petal length vs petal width resulted in r = 0.96 indicating that the petal length and width showed a strong positive correlation and that their measurements often varied together. The Logistic Regression (classification techniques) were used to predict the species of iris flowers based on their features, and the Linear Regression (regression techniques) were used to predict continuous values based on features."
    " For the linear regression (LRM) analysis, petal length vs petal width gave an R^2 value of 0.93, indicating that most of the variance in petal width can be explained by petal length. Some predictions were made for petal width following an 80 - 20 split in the data for training and testing, respectively. The analysis has also shown that the dataset is suitable for classification and regression tasks, and the techniques used have provided valuable insights into the relationships between features and the target variable."
    " The analysis has demonstrated that the features (sepal length/width, petal length/width) are valuable for classification of iris species, and the techniques used have provided valuable insights into the relationships between features and the target variable. However, in order to have a more reliable method for predicting the species using a linear regression (or logistic regression) model, a larger sample population is essential in order to accurately visualise and calculate the nuances between such species based on their features."
    " In terms of model accuracy, reliability, and consistent repeatability the size of the dataset may be considered a limiting factor. However, the data does efficiently demonstrate what a linear based dataset can show through various forms of analysis."
    )
    print("", file=file)
    print(textwrap.fill(conclusion_text, width=210), file=file)

print("Conclusion appended to analysis.md")

# References - add references to the analysis.txt file.

with open("analysis.md", "a") as file:
    print("\n## References", file=file)
    print("\n\tAcademic Sources", file=file)
    print("\nAnderson, E. (1935) ‘The irises of the Gaspé peninsula’, Bulletin of the American Iris Society, 59, pp. 2–5.", file=file)
    print("\nCheeseman, P., Kelly, J., Self, M. and Taylor, W. (1988) AUTOCLASS II conceptual clustering system finds 3 classes in the data, MLC Proceedings, pp. 54–64. Available at: https://cdn.aaai.org/AAAI/1988/AAAI88-108.pdf", file=file)
    print("\nDasarathy, B.V. (1980) ‘Nosing around the neighborhood: a new system structure and classification rule for recognition in partially exposed environments’, IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-2(1), pp. 67–71. Available at: https://www.academia.edu/30910064/Nosing_Around_the_Neighborhood_A_New_System_Structure_and_Classification_Rule_for_Recognition_in_Partially_Exposed_Environments", file=file)
    print("\nDomingos, P. (2012) ‘A few useful things to know about machine learning’, Communications of the ACM, 55(10), pp. 78–87. Available at: https://dl.acm.org/doi/10.1145/2347736.2347755", file=file)
    print("\nDuda, R.O. and Hart, P.E. (1973) Pattern Classification and Scene Analysis. New York: John Wiley & Sons. Available at: https://www.semanticscholar.org/paper/Pattern-classification-and-scene-analysis-Duda-Hart/b07ce649d6f6eb636872527104b0209d3edc8188", file=file)
    print("\nFisher, R.A. (1936) ‘The use of multiple measurements in taxonomic problems’, Annual Eugenics, 7(Part II), pp. 179–188. Available at: https://onlinelibrary.wiley.com/doi/10.1111/j.1469-1809.1936.tb02137.x", file=file)
    print("\nFisher, R.A. (1950) Contributions to Mathematical Statistics. New York: Wiley & Co.", file=file)
    print("\nGates, G.W. (1972) ‘The reduced nearest neighbor rule’, IEEE Transactions on Information Theory, 18(3), pp. 431–433. Available at: https://ieeexplore.ieee.org/document/1054809", file=file)
    print("\nHamilton, K.M. (2022) Drug resistance and susceptibility in sheep nematodes: fitness and the role of anthelmintic combinations in resistance management. PhD Thesis. University College Dublin, Teagasc, and AgResearch.", file=file)
    print("\nJames, G., Witten, D., Hastie, T. and Tibshirani, R. (2013) An Introduction to Statistical Learning. New York: Springer. Available at: https://link.springer.com/book/10.1007/978-1-0716-1418-1", file=file)
    print("\nJolliffe, I.T. and Cadima, J. (2016) ‘Principal component analysis: a review and recent developments’, Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, 374(2065), pp. 20150202. Available at: https://pubmed.ncbi.nlm.nih.gov/26953178/", file=file)
    print("\nKuhn, M. and Johnson, K. (2013) Applied Predictive Modeling. Springer. Available at: https://link.springer.com/book/10.1007/978-1-4614-6849-3", file=file)
    print("\n", file=file)
    print("\n\tInformation Sources (Non-Academic)", file=file)
    print("\nAnalytics Vidhya (2020) ‘Confusion matrix in machine learning’. Available at: https://www.analyticsvidhya.com/blog/2020/04/confusion-matrix-machine-learning/", file=file)
    print("\nAnalytics Vidhya (2024) ‘Pair plots in machine learning’. Available at: https://www.analyticsvidhya.com/blog/2024/02/pair-plots-in-machine-learning/", file=file)
    print("\nBuilt In (no date) ‘Seaborn pairplot’. Available at: https://builtin.com/articles/seaborn-pairplot", file=file)
    print("\nBytemedirk (no date) ‘Mastering iris dataset analysis with Python’. Available at: https://bytemedirk.medium.com/mastering-iris-dataset-analysis-with-python-9e040a088ef4", file=file)
    print("\nDatacamp (no date) ‘Simple linear regression tutorial’. Available at: https://www.datacamp.com/tutorial/simple-linear-regression", file=file)
    print("\nDatatab (no date) ‘Linear regression tutorial’. Available at: https://datatab.net/tutorial/linear-regression", file=file)
    print("\nGeeksforGeeks (no date) ‘Exploratory data analysis on iris dataset’. Available at: https://www.geeksforgeeks.org/exploratory-data-analysis-on-iris-dataset/", file=file)
    print("\nGeeksforGeeks (no date) ‘How to show first/last n rows of a dataframe’. Available at: https://stackoverflow.com/questions/58260771/how-to-show-firstlast-n-rows-of-a-dataframe", file=file)
    print("\nGeeksforGeeks (no date) ‘Iris dataset’. Available at: https://www.geeksforgeeks.org/iris-dataset/", file=file)
    print("\nGeeksforGeeks (no date) ‘Interpretations of histogram’. Available at: https://www.geeksforgeeks.org/interpretations-of-histogram/", file=file)
    print("\nGeeksforGeeks (no date) ‘ML mathematical explanation of RMSE and R-squared error’. Available at: https://www.geeksforgeeks.org/ml-mathematical-explanation-of-rmse-and-r-squared-error/", file=file)
    print("\nGeeksforGeeks (no date) ‘Python basics of pandas using iris dataset’. Available at: https://www.geeksforgeeks.org/python-basics-of-pandas-using-iris-dataset/", file=file)
    print("\nGist (no date) ‘Iris dataset CSV’. Available at: https://gist.githubusercontent.com/", file=file)
    print("\nHow.dev (no date) ‘How to perform the ANOVA test in Python’. Available at: https://how.dev/answers/how-to-perform-the-anova-test-in-python", file=file)
    print("\nIBM (no date) ‘Introduction to linear discriminant analysis’. Available at: https://www.ibm.com/think/topics/linear-discriminant-analysis", file=file)
    print("\nIBM (no date) ‘Linear regression’. Available at: https://www.ibm.com/think/topics/linear-regression", file=file)
    print("\nIBM (no date) ‘Logistic regression’. Available at: https://www.ibm.com/think/topics/logistic-regression", file=file)
    print("\nInvestopedia (no date) ‘R-squared’. Available at: https://www.investopedia.com/terms/r/r-squared.asp", file=file)
    print("\nKachiann (no date) ‘A beginners guide to machine learning with Python: Iris flower prediction’. Available at: https://medium.com/@kachiann/a-beginners-guide-to-machine-learning-with-python-iris-flower-prediction-61814e095268", file=file)
    print("\nKulkarni, M. (no date) ‘Heatmap analysis using Python seaborn and matplotlib’. Available at: https://medium.com/@kulkarni.madhwaraj/heatmap-analysis-using-python-seaborn-and-matplotlib-f6f5d7da2f64", file=file)
    print("\nMedium (no date) ‘Exploratory data analysis of iris dataset’. Available at: https://medium.com/@nirajan.acharya777/exploratory-data-analysis-of-iris-dataset-9c0df76771df", file=file)
    print("\nMedium (no date) ‘Pairplot visualization’. Available at: https://medium.com/analytics-vidhya/pairplot-visualization-16325cd725e6", file=file)
    print("\nMedium (no date) ‘Regression model evaluation metrics’. Available at: https://medium.com/%40brandon93.w/regression-model-evaluation-metrics-r-squared-adjusted-r-squared-mse-rmse-and-mae-24dcc0e4cbd3", file=file)
    print("\nMedium (2023) 'Scikit-learn, the iris dataset, and machine learning: the journey to a new skill'. Medium. Available at: https://3tw.medium.com/scikit-learn-the-iris-dataset-and-machine-learning-the-journey-to-a-new-skill-c8d2f537e087", file=file)
    print("\nMizanur (no date) ‘Cleaning your data: handling missing and duplicate values’. Available at: https://mizanur.io/cleaning-your-data-handling-missing-and-duplicate-values/", file=file)
    print("\nNewcastle University (no date) ‘Box and whisker plots’. Available at: https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/statistics/data-presentation/box-and-whisker-plots.html", file=file)
    print("\nNick McCullum (no date) ‘Python visualization: boxplot’. Available at: https://www.nickmccullum.com/python-visualization/boxplot/", file=file)
    print("\nNumpy (no date) ‘numpy.polyfit’. Available at: https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html", file=file)
    print("\nPandas (no date) ‘pandas.read_csv’. Available at: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html", file=file)
    print("\nPython Documentation (no date) ‘Built-in types’. Available at: https://docs.python.org/3/library/stdtypes.html", file=file)
    print("\nResearchGate (no date) ‘Classification of Iris Flower Dataset using Different Algorithms’. Available at: https://www.researchgate.net/publication/367220930_Classification_of_Iris_Flower_Dataset_using_Different_Algorithms", file=file)
    print("\nRSS (no date) ‘Common statistical terms’. Available at: https://rss.org.uk/resources/statistical-explainers/common-statistical-terms/", file=file)
    print("\nScikit-learn (no date) ‘Classification report’. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html", file=file)
    print("\nScikit-learn (no date) ‘LabelEncoder’. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html", file=file)
    print("\nScikit-learn (no date) ‘LinearRegression’. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html", file=file)
    print("\nScikit-learn (no date) ‘LogisticRegression’. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html", file=file)
    print("\nScikit-learn (no date) ‘PCA example with iris dataset’. Available at: https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html", file=file)
    print("\nScikit-learn Documentation (2021) ‘Plot Iris Dataset Example’. Available at: https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html", file=file)
    print("\nSeaborn (no date) ‘Pairplot’. Available at: https://seaborn.pydata.org/generated/seaborn.pairplot.html", file=file)
    print("\nSeaborn (no date) ‘Regplot’. Available at: https://seaborn.pydata.org/generated/seaborn.regplot.html", file=file)
    print("\nSeaborn (no date) ‘Scatterplot’. Available at: https://seaborn.pydata.org/generated/seaborn.scatterplot.html", file=file)
    print("\nSlidescope (no date) ‘ANOVA example using Python pandas on iris dataset’. Available at: https://slidescope.com/anova-example-using-python-pandas-on-iris-dataset/#:~:text=We%20then%20convert%20the%20dataset,p-value%20for%20the%20test", file=file)
    print("\nStack Overflow (no date) ‘How to show first/last n rows of a dataframe’. Available at: https://stackoverflow.com/questions/58260771/how-to-show-firstlast-n-rows-of-a-dataframe", file=file)
    print("\nToxigon (no date) ‘Best practices for data cleaning and preprocessing’. Available at: https://toxigon.com/best-practices-for-data-cleaning-and-preprocessing", file=file)
    print("\nToxigon (no date) ‘Guide to data cleaning’. Available at: https://toxigon.com/guide-to-data-cleaning", file=file)
    print("\nToxigon (no date) ‘Introduction to seaborn for data visualization’. Available at: https://toxigon.com/introduction-to-seaborn-for-data-visualization", file=file)
    print("\nToxigon (no date) ‘Seaborn data visualization guide’. Available at: https://toxigon.com/seaborn-data-visualization-guide", file=file)
    print("\nUCI Machine Learning Repository (2025) ‘Iris Dataset’. Available at: https://archive.ics.uci.edu/dataset/53/iris", file=file)
    print("\nWV State University (no date) ‘Scholarly vs. non-scholarly articles’. Available at: https://wvstateu.libguides.com/c.php?g=813217&p=5816022", file=file)
    print("\nWikipedia (no date) ‘Linear discriminant analysis’. Available at: https://en.wikipedia.org/wiki/Linear_discriminant_analysis", file=file)

print("References appended to analysis.md")

with open("analysis.md", "a") as file:
    print("\n# END OF ANALYSIS DOCUMENT", file=file)

print("End of Document appended to analysis.md")

# END
