# Code Explained

*author: Kyra Menai Hamilton*

This document explains in more detail what each part of the code does and why it is used.

## Packages

The packages used for this analysis and saving and exporting of files is as follows:

```ruby
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
```

## Data import

Sourcing the dataframe from the UCI Repository.

```ruby
iris = fetch_ucirepo(id=53)
# data - extracting x and y (as pandas dataframes)
x = iris.data.features
y = iris.data.targets

print(iris.metadata) # metadata - print was to check
print(iris.variables) # variable information - print was to check
```

The features and targets needed to be combined into a single dataframe so that it could be exported as a CSV file.

```ruby
iris_df = pd.concat([x, y], axis=1)
```

Exporting the dataframe into a CSV file to make the dataset more easily accessible and useful.

```ruby
iris_df.to_csv
   ('D:/Data_Analytics/Modules/PandS/pands-project/iris.csv', index=False)
 print("Iris dataset has been successfully exported to a CSV!")
# Output - Iris dataset has been successfully exported to a CSV!
```

For retrieving dataset to making plots and for analysis the dataset needed to be imported into Python from the CSV file.

```ruby
iris_df = pd.read_csv('D:/Data_Analytics/Modules/PandS/pands-project/iris.csv')
print(iris_df)
```

## EDA

Summary statistics for the whole dataset and for each species was done using the ```df.describe()``` function.
This was modified for each species so that the species could be separated from one another:

```ruby
    setosa_stats = iris_df[iris_df['class'] == 'Iris-setosa'].describe()
    versicolor_stats = iris_df[iris_df['class'] == 'Iris-versicolor'].describe()
    virginica_stats = iris_df[iris_df['class'] == 'Iris-virginica'].describe()
```

Class distributions were also explored using ```iris_df['class'].value_counts()``` to see if the data was evenly distributed between the three species. 50 samples were seen for each of the species, Setosa, Virginica, and Versicolor.

## Boxplots

Boxplots were plotted for each of the four species and the data within colour coded by species.
These were plotted using ```sns.boxplot(x='class', y=feature, hue='class', data=iris_df, ax=ax)```, and plot size was determined by ```plt.figure(figsize=(12, 8))```.

- ```sns.boxplot``` refers to the plot to be made,
- ```for i, feature in enumerate(features):``` separates by feature and labels the figure appropriately in ```ax.set_title(titles[i])```,
- where ```data=iris_df``` was the iris dataframe,
- ```x='class'``` where the data will be separated by species,
- the ```y=feature``` is plotted against the species
- ```hue="class"``` he data will be colour coded by species, and
- ```ax=ax``` referred to the subplot.

## Histograms

Histograms were plotted for each of the features.
All plots were put into one "file" saved as a PNG to make the data easier to read and compare using ```fig, axes = plt.subplots(2, 2, figsize=(12, 10))```.
Each of the four histograms was plotted using ```sns.histplot(data=iris_df, x="feature", hue="class", kde=False, ax=axes[0, 0], bins=15)```.
Similarly to the boxplots the breakdown of the code is as follows:

- ```sns.histplot``` refers to the plot to be made,
- where ```data=iris_df``` was the iris dataframe,
- ```x="feature"``` where ```"feature"``` was sepal length/width or petal length/width,
- ```hue="class"``` would colour code the plot points by species (referred to in the dataframe as class), and
- ```ax=axes[0, 0]``` referred to the subplot.

## Scatterplots

Both scatterplots were put into one "file" saved as a PNG to make the data easier to read and compare using ```fig, axes = plt.subplots(1, 2, figsize=(20, 8))```.

Both scatterplots were plotted using the code ```sns.scatterplot(ax=axes[0], data=iris_df, x='feature1', y='feature2', hue='class', s=100)```.

- where ```sns.scatterplot``` refers to the plot to be run,
- ```ax=axes[0]``` refers to the subplot,
- ```data=df``` was the iris dataframe,
- ```x="feature1"``` where ```"feature1"``` was sepal length or petal length,
- ```y="feature2"``` where ```"feature2"``` was sepal width or petal width, and
- ```hue="class"``` colours the plot points by species.

## Pairplots

For the Pairwise plot, the pairplot from the package ```seaborn``` was used:

```ruby
import seaborn as sns
pairplot = sns.pairplot(dataframe, hue='class', height=desired height for plot) 
```

- Where ```dataframe``` is the data to be plotted,
- ```class``` is the categorical variable, and
- ```height``` is the desired height of the subplot.

## Correlation Matrix

```ruby
corr_matrix = df.iloc[:, :4].corr() :
```

- where ```df.iloc[:, :4]:``` selects the first four columns of the df (dataframe), these were assumed to be numerical features (sepal length, sepal width, petal length, and petal width).
- in this code, ```:``` selects all rows, and ```:4``` selects columns from index 0 to 3 (exclusive).
- ```.corr():``` calculates the correlation matrix for the selected columns.

The correlation matrix shows the pairwise correlation coefficients between the features, with values ranging from:

-  1: Perfect positive correlation.
-  0: No correlation.
- -1: Perfect negative correlation.

```ruby
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm') :
```

- where ```sns.heatmap(corr_matrix, ...):``` is what creates a heatmap visualisation of the correlation matrix using Seaborn.
- ```annot=True:``` shows the correlation values (numerical) inside each cell of the heatmap.
- ```cmap='coolwarm':``` specifies the colour map for the heatmap.

In the plot,

- cool colours (**blues**) represent **negative** correlations, and
- warm colours (**reds**) represent **positive** correlations.

## PCA

Principal Component Analysis (PCS) was conducted on the data. The data needed to be standardised for the analysis and this was done through a scaled transformation of the data:

```ruby
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## K-means

K-means cluster analysis to look for underlying patterns or clustering within the data.

```ruby
kmeans = KMeans(n_clusters=3, random_state=42)
iris_df['cluster'] = kmeans.fit_predict(iris_df.iloc[:, :4])
```

- ```kmeans``` is the test to be run,
- ```n_clusters = 3``` refers to the number of clusters (the 3 species)
- ```iris_df['cluster']``` instructs to split the dataset into clusters, and
- ```kmeans.fit_predict(iris_df.iloc[:, :4])``` asks to fit the first 4 columns (features).

## LRM

The packages used to conduct the Logistic Regression and get R<sup>2</sup> were:

```ruby
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
```

All plots (two subplots for sepal features and petal features) were placed into one "figure" using ```fig, axes = plt.subplots(1, 2, figsize=(20, 8))```.
The input code has been split for easier explanation.

```ruby
# Feature1 vs Feature2
X_feature = iris_df[['feature1']]
y_feature = iris_df['feature2']
model_feature = LinearRegression()
model_feature.fit(X_feature, y_feature)
y_feature_pred = model_feature.predict(X_feature)
r2_feature = r2_score(y_feature, y_feature_pred)
```

- where ```X_feature``` refers to the input variable ```iris_df[['feature1']]``` which would be sepal length or petal length,
- ```y_feature``` refers to the input variable ```iris_df['feature2']``` which would be sepal width or petal width,
- ```model_feature = LinearRegression()``` refers to the model that is being run in this case it is ```LinearRegression()```,
- ```model_feature.fit(X_feature, y_feature``` states the model to be fit,
- ```y_feature_pred = model_feature.predict(X_feature)``` where ```y_feature_pred``` was sepal width or petal width to be predicted from the ```(X_feature)``` of sepal length or petal length, and
- ```r2_feature = r2_score(y_feature, y_feature_pred)``` would give the ```r2_feature``` of the model based on the actual ```y_feature``` compared to the ```y_feature_pred``` values.

```ruby
sns.scatterplot(ax=axes[0], data=iris_df, x='feature1', y='feature2', hue='species', s=100)
sns.regplot(ax=axes[0], data=iris_df, x='feature1', y='feature2', scatter=False, color='red')
axes[0].set_title('Feature1 vs Feature2 by Species')
axes[0].set_xlabel('Feature1 (cm)')
axes[0].set_ylabel('Feature2 (cm)')
axes[0].legend(title='Species')
axes[0].grid(True)
axes[0].text(0.05, 0.95, f'R² = {r2_feature:.2f}', transform=axes[0].transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
```

- where ```sns.scatterplot``` refers to the plot to be run,
- where ```sns.regplot``` refers to the regression line,
- ```ax=axes[0]``` refers to the subplot,
- ```data=df``` was the iris dataframe,
- ```x="feature1"``` where ```"feature1"``` was sepal length or petal length,
- ```y="feature2"``` where ```"feature2"``` was sepal width or petal width, and
- ```hue="class"``` would colour code the plot points by species.

## Log

```ruby
X_species = iris_df[['sepal length', 'sepal width', 'petal length', 'petal width']]
y_species = iris_df['class']
le = LabelEncoder()
y_species = le.fit_transform(y_species) # setosa = 0, versicolor = 1, virginica = 2
X_species_train, X_species_test, y_species_train, y_species_test = train_test_split(X_species, y_species, test_size=0.2, random_state=42)

# Create and fit the model
model_species = LogisticRegression(max_iter=200)
model_species.fit(X_species_train, y_species_train)
y_species_pred = model_species.predict(X_species_test)
accuracy = accuracy_score(y_species_test, y_species_pred)
```

- ```X_species``` is the features within the dataset (sepal length.width, petal length/width),
- the ```y_species``` is the species (class) of Iris flower,
- in ```y_species = le.fit_transform(y_species)``` the ```y_species``` is transformed into numerical values (Setosa = 0, Versicolor = 1, Virginica = 2).
- ```X_species_train, X_species_test, y_species_train, y_species_test = train_test_split(X_species, y_species, test_size=0.2, random_state=42)``` splits the data into test (0.2 / 20 %) and train data,
- the model was fitted using ```model_species.fit(X_species_train, y_species_train)```, and
- the model accuracy was checked using ```accuracy = accuracy_score(y_species_test, y_species_pred)```.

## Confusion

A confusion matrix was conducted ```cm = confusion_matrix(y_species_test, y_species_pred)``` and plotted to visualise the results of the Logistic Regression using:

```ruby
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', 
            xticklabels=le.classes_, yticklabels=le.classes_)
```

- ```sns.heatmap``` refers to the plot to be made,
- ```cm``` is what is to be plotted,
- ```cmap='Blues'``` gives the colours for the heatmap, and
- ```xticklabels=le.classes_``` and ```yticklabels=le.classes_``` are the predicted (x) and actual (y) classes plotted against one another.

# END OF CODE EXPLAINED