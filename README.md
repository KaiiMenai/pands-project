# Project for the Programming and Scripting Module

## Background

The iris dataset (originally sourced by Anderson in 1935) is a well examined and thoroughly evaluated dataset of three species of iris flower (Setosa, Versicolor, Virginica) totalling 150 samples measuring four key features (sepal length/width, petal length/width) that aid in differentiation between the species. Not only highlighting the importance of field expert knowledge when designing a research project and its analysis, but also highlighting that there can be overlap between different species within the same genus (highlighted by the repeated overlap for Versicolor and Virginica for some features). In 1936, Fisher used the dataset as an example of linear discriminant analysis, used for finding a linear combination of features that chan characterise or separate classes or objects (specifically two or more). Throughout the decades, the iris dataset has been used thoroughly as an example dataset for machine learning and algorithm development and is stored as an open access dataset in the UCI Machine Learning Repository.

## Overview of project

This project focused on using the data in a number of different statistical tests to see what patterns were visible in the data, whether there were indeed differences between the species, and what pairs of features had influence over the other.

Initial **exploratory data analysis** described the shape of the dataset (150 samples 5 variables). A check on **first and last five rows** of the dataset was conduced, and the presence of missing values was checked (none were found). The **summary statistics** were initially conducted on the dataset as a whole, before being split by species for a more granular analysis. For the overall **summary statistics** it was found that petal length had the largest **standard deviation** (difference from mean value) of 1.76 respectively, suggesting that they varied more widely across samples compared to sepal features, indicating that maybe this would be useful feature to investigate further to see where these differences were.

**Boxplots** and **histograms** were plotted for each of the four features with the samples separated by species. **Boxplots** helped with finding outliers in the data, whilst the Histograms showed the frequency of specific measurements for each feature. Both of these plots highlighted that the measurements for the Setosa species of iris flower separated from the other two species for all four of the features. Versicolor and Virginica overlap considerably for the sepal features (length/width), however, species separation was seen for all 3 species for the petal features (length/width) suggesting that these would be useful features in differentiating between species. The **scatterplots** showed similar results.

**Pairwise comparison (pairplot)** was also conducted on the data in order do get an overall quick analysis of the relationship between different pairs of features. Petal length vs. petal width again showed the most distinct clustering of the data by species, adding weight to the theory that the petal features would be useful in differentiating between iris species. This is further highlighted by the **correlation matrix** where a strong positive correlation is shown between petal length and petal width ( r = 0.96).

To further analyse the variation within the data and to see where most of the differences between features for the species originates **principal component analysis (PCA)**. It was found that most variance within the data was seen between the petal features (the first principal components (PC1)). K-means clustering was conducted to check if there were indeed patterns within the iris dataset.

**Linear regression** was used to test this further. Looking at feature vs feature and then also separating the analysis on a more granular scale to look specifically at the species results. Petal features have significant influence over one another (R<sup>2</sup> = 0.93).

Finally **logistic regression** was conducted on the dataset to see whether it could be used to predict a species of iris flower based purely on inputted values. The outcome for the accuracy of this potential model was 1.0, a perfect model, giving a high level of accuracy for any predictions made. This accuracy result is likely due to the simplicity and linear separability of the iris dataset. A **confusion matrix** was plotted to aid in visualising the performance of the **logistic regression model**, aiding in seeing clearly where discrepancies would be in the species predicted.

## Conclusion

Assessing species-feature relationships aided in understanding where in the dataset the majority of the variance came from, which features had influence over one another, and which of the measured features would be the most useful as tools for differentiation. A number of these answers are the same. Most of the variance in the dataset comes from the petal length and petal width features, the petal length and width features also influenced one another, and these features were also shown to be the most useful in differentiating between and predicting the species of iris flower.

## Explain what the code does for the important tests

### Packages

### Data import

```ruby
iris = fetch_ucirepo(id=53)
# data - extracting x and y (as pandas dataframes)
x = iris.data.features
y = iris.data.targets

print(iris.metadata) # metadata - print was to check
print(iris.variables) # variable information - print was to check
```

Combine the features and targets into a single df to export as CSV
```ruby
iris_df = pd.concat([x, y], axis=1)
```

Exporting the DataFrame (df) to a CSV file
```ruby
iris_df.to_csv
   ('D:/Data_Analytics/Modules/PandS/pands-project/iris.csv', index=False)
 print("Iris dataset has been successfully exported to a CSV!")
# Output - Iris dataset has been successfully exported to a CSV!
```

Now to retrieve dataset for making plots and analysis
Import the dataset from the CSV file

```ruby
iris_df = pd.read_csv('D:/Data_Analytics/Modules/PandS/pands-project/iris.csv')
print(iris_df)
```

### EDA

Summary statistics for the whole dataset and for each species was done using the ```df.describe()``` function. 
This was modified for each species so that the species could be separated from one another:

```ruby
    setosa_stats = iris_df[iris_df['class'] == 'Iris-setosa'].describe()
    versicolor_stats = iris_df[iris_df['class'] == 'Iris-versicolor'].describe()
    virginica_stats = iris_df[iris_df['class'] == 'Iris-virginica'].describe()
```

Class distributions were also explored using ```df['species'].value_counts()``` to see if the data was evenly distributed between the three species. 50 samples were seen for each of the species, Setosa, Virginica, and Versicolor.

### Boxplots

Boxplots were plotted for each of the four species and the data within colour coded by species.
These were plotted using ```sns.boxplot(x='species', y=feature, hue='species', data=df, ax=ax)```, and plot size was determined by ```plt.figure(figsize=(12, 8))```.

- ```sns.boxplot``` refers to the plot to be made,
- ```for i, feature in enumerate(features):``` separates by feature and labels the figure appropriately in ```ax.set_title(titles[i])```,
- where ```data=iris_df``` was the iris dataframe,
- ```x='species'``` where the data will be separated by species,
- the ```y=feature``` is plotted against the species
- ```hue="class"``` he data will be colour coded by species, and
- ```ax=ax``` referred to the subplot.

### Histograms

Histograms were plotted for each of the features.
All plots were put into one "figure" saved as a PNG to make the data easier to read and compare using ```fig, axes = plt.subplots(2, 2, figsize=(12, 10))```.
Each of the four histograms was plotted using ```sns.histplot(data=iris_df, x="feature", hue="class", kde=False, ax=axes[0, 0], bins=15)```.
Similarly to the boxplots the breakdown of the code is as follows:

- ```sns.histplot``` refers to the plot to be made,
- where ```data=iris_df``` was the iris dataframe,
- ```x="feature"``` where ```"feature"``` was sepal length/width or petal length/width,
- ```hue="class"``` would colour code the plot points by species (referred to in the dataframe as class), and
- ```ax=axes[0, 0]``` referred to the subplot.

### Scatterplots

### Pairplots

For the Pairwise plot, the pairplot from the package ```seaborn``` was used:

```ruby
import seaborn as sns
pairplot = sns.pairplot(dataframe, hue='class', height=desired height for plot) 
```

- Where ```dataframe``` is the data to be plotted,
- ```class``` is the categorical variable, and
- ```height``` is the desired height of the subplot.

### Correlation Matrix

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

### PCA

Principal Component Analysis (PCS) was conducted on the data. The data needed to be standardised for the analysis and this was done through a scaled transformation of the data:

```ruby
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### K-means

### LRM

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
- ```Xyfeature``` refers to the input variable ```iris_df['feature2']``` which would be sepal width or petal width,
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

### Log

### Confusion

## References

### Academic Sources

Anderson, E. (1935). *The irises of the Gaspé peninsula*, Bulletin of the American Iris Society, 59, pp. 2–5.

Cheeseman, P., Kelly, J., Self, M. and Taylor, W. (1988). *AUTOCLASS II conceptual clustering system finds 3 classes in the data*, MLC Proceedings, pp. 54–64. Available at: https://cdn.aaai.org/AAAI/1988/AAAI88-108.pdf

Dasarathy, B.V. (1980). *Nosing around the neighborhood: a new system structure and classification rule for recognition in partially exposed environments*, IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-2(1), pp. 67–71. Available at: https://www.academia.edu/30910064/Nosing_Around_the_Neighborhood_A_New_System_Structure_and_Classification_Rule_for_Recognition_in_Partially_Exposed_Environments

Domingos, P. (2012). *A few useful things to know about machine learning*, Communications of the ACM, 55(10), pp. 78–87. Available at: https://dl.acm.org/doi/10.1145/2347736.2347755

Duda, R.O. and Hart, P.E. (1973). *Pattern Classification and Scene Analysis*. New York: John Wiley & Sons. Available at: https://www.semanticscholar.org/paper/Pattern-classification-and-scene-analysis-Duda-Hart/b07ce649d6f6eb636872527104b0209d3edc8188

Fisher, R.A. (1936). *The use of multiple measurements in taxonomic problems*, Annual Eugenics, 7(Part II), pp. 179–188. Available at: https://onlinelibrary.wiley.com/doi/10.1111/j.1469-1809.1936.tb02137.x

Fisher, R.A. (1950). *Contributions to Mathematical Statistics*. New York: Wiley & Co.

Gates, G.W. (1972). *The reduced nearest neighbor rule*, IEEE Transactions on Information Theory, 18(3), pp. 431–433. Available at: https://ieeexplore.ieee.org/document/1054809

Hamilton, K.M. (2022). *Drug resistance and susceptibility in sheep nematodes: fitness and the role of anthelmintic combinations in resistance management*. PhD Thesis. University College Dublin, Teagasc, and AgResearch.

James, G., Witten, D., Hastie, T. and Tibshirani, R. (2013). *An Introduction to Statistical Learning*. New York: Springer. Available at: https://link.springer.com/book/10.1007/978-1-0716-1418-1

Jolliffe, I.T. and Cadima, J. (2016). *Principal component analysis: a review and recent developments*, Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, 374(2065), pp. 20150202. Available at: https://pubmed.ncbi.nlm.nih.gov/26953178/

Kuhn, M. and Johnson, K. (2013). *Applied Predictive Modeling*. Springer. Available at: https://link.springer.com/book/10.1007/978-1-4614-6849-3

### Information Sources (Non-Academic)

Analytics Vidhya. (2020). *Confusion matrix in machine learning*. Available at: https://www.analyticsvidhya.com/blog/2020/04/confusion-matrix-machine-learning/

Analytics Vidhya. (2024). *Pair plots in machine learning*. Available at: https://www.analyticsvidhya.com/blog/2024/02/pair-plots-in-machine-learning/

Built In. (no date). *Seaborn pairplot*. Available at: https://builtin.com/articles/seaborn-pairplot

Bytemedirk. (no date). *Mastering iris dataset analysis with Python*. Available at: https://bytemedirk.medium.com/mastering-iris-dataset-analysis-with-python-9e040a088ef4

Datacamp. (no date). *Simple linear regression tutorial*. Available at: https://www.datacamp.com/tutorial/simple-linear-regression

Datatab. (no date). *Linear regression tutorial*. Available at: https://datatab.net/tutorial/linear-regression

GeeksforGeeks. (no date). *Exploratory data analysis on iris dataset*. Available at: https://www.geeksforgeeks.org/exploratory-data-analysis-on-iris-dataset/

GeeksforGeeks. (no date). *How to show first/last n rows of a dataframe*. Available at: https://stackoverflow.com/questions/58260771/how-to-show-firstlast-n-rows-of-a-dataframe

GeeksforGeeks. (no date). *Iris dataset*. Available at: https://www.geeksforgeeks.org/iris-dataset/

GeeksforGeeks. (no date). *Interpretations of histogram*. Available at: https://www.geeksforgeeks.org/interpretations-of-histogram/

GeeksforGeeks. (no date). *ML mathematical explanation of RMSE and R-squared error*. Available at: https://www.geeksforgeeks.org/ml-mathematical-explanation-of-rmse-and-r-squared-error/

GeeksforGeeks. (no date). *Python basics of pandas using iris dataset*. Available at: https://www.geeksforgeeks.org/python-basics-of-pandas-using-iris-dataset/

Gist. (no date). *Iris dataset CSV*. Available at: https://gist.githubusercontent.com/

How.dev. (no date). *How to perform the ANOVA test in Python*. Available at: https://how.dev/answers/how-to-perform-the-anova-test-in-python

IBM. (no date). *Introduction to linear discriminant analysis*. Available at: https://www.ibm.com/think/topics/linear-discriminant-analysis

IBM. (no date). *Linear regression*. Available at: https://www.ibm.com/think/topics/linear-regression

IBM. (no date). *Logistic regression*. Available at: https://www.ibm.com/think/topics/logistic-regression

Investopedia. (no date). *R-squared*. Available at: https://www.investopedia.com/terms/r/r-squared.asp

Kachiann. (no date). *A beginners guide to machine learning with Python: Iris flower prediction*. Available at: https://medium.com/@kachiann/a-beginners-guide-to-machine-learning-with-python-iris-flower-prediction-61814e095268

Kulkarni, M. (no date). *Heatmap analysis using Python seaborn and matplotlib*. Available at: https://medium.com/@kulkarni.madhwaraj/heatmap-analysis-using-python-seaborn-and-matplotlib-f6f5d7da2f64

Medium. (no date). *Exploratory data analysis of iris dataset*. Available at: https://medium.com/@nirajan.acharya777/exploratory-data-analysis-of-iris-dataset-9c0df76771df

Medium. (no date). *Pairplot visualization*. Available at: https://medium.com/analytics-vidhya/pairplot-visualization-16325cd725e6

Medium. (no date). *Regression model evaluation metrics*. Available at: https://medium.com/%40brandon93.w/regression-model-evaluation-metrics-r-squared-adjusted-r-squared-mse-rmse-and-mae-24dcc0e4cbd3

Medium. (2023). *Scikit-learn, the iris dataset, and machine learning: the journey to a new skill*. Medium. Available at: https://3tw.medium.com/scikit-learn-the-iris-dataset-and-machine-learning-the-journey-to-a-new-skill-c8d2f537e087

Mizanur. (no date). *Cleaning your data: handling missing and duplicate values*. Available at: https://mizanur.io/cleaning-your-data-handling-missing-and-duplicate-values/

Newcastle University. (no date). *Box and whisker plots*. Available at: https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/statistics/data-presentation/box-and-whisker-plots.html

Nick McCullum. (no date). *Python visualization: boxplot*. Available at: https://www.nickmccullum.com/python-visualization/boxplot/

Numpy. (no date). *numpy.polyfit*. Available at: https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html

Pandas. (no date). *pandas.read_csv*. Available at: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

Python Documentation. (no date). *Built-in Types*. Available at: https://docs.python.org/3/library/stdtypes.html

ResearchGate. (no date). *Classification of Iris Flower Dataset using Different Algorithms*. Available at: https://www.researchgate.net/publication/367220930_Classification_of_Iris_Flower_Dataset_using_Different_Algorithms

RSS. (no date). *Common statistical terms*. Available at: https://rss.org.uk/resources/statistical-explainers/common-statistical-terms/

Scikit-learn. (no date). *Classification report*. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html

Scikit-learn. (no date). *LabelEncoder*. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

Scikit-learn. (no date). *LinearRegression*. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

Scikit-learn. (no date). *LogisticRegression*. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

Scikit-learn. (no date). *PCA example with iris dataset*. Available at: https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html

Scikit-learn Documentation. (2021). *Plot Iris Dataset Example*. Available at: https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html

Seaborn. (no date). *Pairplot*. Available at: https://seaborn.pydata.org/generated/seaborn.pairplot.html

Seaborn. (no date). *Regplot*. Available at: https://seaborn.pydata.org/generated/seaborn.regplot.html

Seaborn. (no date). *Scatterplot*. Available at: https://seaborn.pydata.org/generated/seaborn.scatterplot.html

Slidescope. (no date). *ANOVA example using Python pandas on iris dataset*. Available at: https://slidescope.com/anova-example-using-python-pandas-on-iris-dataset/#:~:text=We%20then%20convert%20the%20dataset,p-value%20for%20the%20test

Stack Overflow. (no date). *How to show first/last n rows of a dataframe*. Available at: https://stackoverflow.com/questions/58260771/how-to-show-firstlast-n-rows-of-a-dataframe

Toxigon. (no date). *Best practices for data cleaning and preprocessing*. Available at: https://toxigon.com/best-practices-for-data-cleaning-and-preprocessing

Toxigon. (no date). *Guide to data cleaning*. Available at: https://toxigon.com/guide-to-data-cleaning

Toxigon. (no date). *Introduction to seaborn for data visualization*. Available at: https://toxigon.com/introduction-to-seaborn-for-data-visualization

Toxigon. (no date). *Seaborn data visualization guide*. Available at: https://toxigon.com/seaborn-data-visualization-guide

UCI Machine Learning Repository. (2025). *Iris Dataset*. Available at: https://archive.ics.uci.edu/dataset/53/iris

WV State University. (no date). *Scholarly vs. non-scholarly articles*. Available at: https://wvstateu.libguides.com/c.php?g=813217&p=5816022

Wikipedia. (no date). *Linear discriminant analysis*. Available at: https://en.wikipedia.org/wiki/Linear_discriminant_analysis