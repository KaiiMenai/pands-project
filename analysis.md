# Data Analysis conducted on the Iris Dataset

## Summary

The analysis used as a source for this document (analysis.py) was conducted using Python and the Pandas, Numpy, Matplotlib, Seaborn, and Scikit-learn libraries. The dataset was imported from the UCI Machine
Learning Repository and is a well-known dataset for classification tasks. The dataset contained 150 samples of iris flowers, with 5 columns: sepal length, sepal width, petal length, petal width, and species
(class). The dataset was used to conduct exploratory data analysis (EDA) and visualisation, as well as some machine learning tasks. Histogram, boxplot, scatterplot, PCA, pairplot, K-means clustering,
correlation matrix, and linear regression analysis were conducted on the dataset, and the results of each were saved as a PNG file.

*Please Note: Some observations taken from my own work for the Principles of Data Analytics Module, with wording changed to match what was required for this project. Link: https://github.com/KaiiMenai/poda-tasks/blob/main/tasks.ipynb*

## Background

Originally sourced by Anderson (1935), the Iris dataset has been used numerous times, with several different iterations available online. Some of these sources contain differing (and often noted as incorrect)
data points, as noted in the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/53/iris). The dataset contained 150 samples of Iris flower, each with five noted variables, four numeric (sepal
and petal feature measurements), and one categorical (the three species), giving a total of 750 sample points throughout the entire dataset. Fisher (1936) made the Iris dataset well known through his paper
titled 'The Use of Multiple Measurements in Taxonomic Problems.' In the paper, the multivariate Iris dataset was used as an example of linear discriminant analysis. That is, a statistical method used to find a
linear combination of features that can either characterise or separate two or more classes of objects or events (https://en.wikipedia.org/wiki/Linear_discriminant_analysis;
https://www.ibm.com/think/topics/linear-discriminant-analysis).

Anderson (1935) originally collected the iris samples to study species variation and hybidisation. Anderson (1935) used the dataset to quantify the morphological differences and variation between Iris species,
focussing on the evolution of the Versicolor species, and how it may have come about as a hybrid of the Setosa and Virginica Iris species. An interesting point about the dataset is that two of the three
species, Iris Versicolor and Iris Virginica, were collected from the same pasture, on the same day, and measured using the same equipment. This is noteworthy for analysis, as Virginica and Versicolor often
appear to converge and are not as easily separable as the Setosa species (histograms, scatter plots, etc.). The Iris dataset has been extensively used as a training dataset, a learning dataset, and for
developing machine learning techniques. The scikit-learn library in Python uses the Iris dataset for demonstration purposes and explains how algorithms can learn from data samples (features) to predict class
labels (https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html; https://archive.ics.uci.edu/dataset/53/iris).

The Iris dataset also highlights the distinction between supervised and unsupervised learning. Supervised learning uses labeled data to train models for classification or regression, while unsupervised learning
explores patterns or clusters in unlabeled data (https://www.ibm.com/think/topics/linear-discriminant-analysis). The iris dataset gives an example of supervised learning problems, particularly multi-class
classification, where the goal is to predict an iris flower's species based on its measurements. Although the dataset only includes flower measurements (sepal length/width and petal length/width) and not
measurements for the entire plant, this focus reflects the expert judgment of Fisher and Anderson, who selected petal and sepal dimensions as meaningful discriminative features. This in turn highlights the
importance of domain expertise in data preparation and model design, suggesting that more efficient analysis and models are built when field experts are involved (Domingos, 2012; Kuhn, and Johnson, 2013).
Despite its age and simplicity, the Iris dataset remains a central teaching tool for introducing classification problems in machine learning.

## Exploratory Data Analysis

The shape of the dataset:
(150, 5)

The dataset contains 150 rows of data and 5 columns. The 5 columns are the species of isis flower (here noted as 'class'), and sepal length, sepal width, petal length, and petal width.

The first and last five rows of the dataset are printed below, as well as the column names within the dataset.

The first 5 rows of the dataset:
|    |   sepal length |   sepal width |   petal length |   petal width | class       |
|---:|---------------:|--------------:|---------------:|--------------:|:------------|
|  0 |            5.1 |           3.5 |            1.4 |           0.2 | Iris-setosa |
|  1 |            4.9 |           3   |            1.4 |           0.2 | Iris-setosa |
|  2 |            4.7 |           3.2 |            1.3 |           0.2 | Iris-setosa |
|  3 |            4.6 |           3.1 |            1.5 |           0.2 | Iris-setosa |
|  4 |            5   |           3.6 |            1.4 |           0.2 | Iris-setosa |

The last 5 rows of the dataset:
|     |   sepal length |   sepal width |   petal length |   petal width | class          |
|----:|---------------:|--------------:|---------------:|--------------:|:---------------|
| 145 |            6.7 |           3   |            5.2 |           2.3 | Iris-virginica |
| 146 |            6.3 |           2.5 |            5   |           1.9 | Iris-virginica |
| 147 |            6.5 |           3   |            5.2 |           2   | Iris-virginica |
| 148 |            6.2 |           3.4 |            5.4 |           2.3 | Iris-virginica |
| 149 |            5.9 |           3   |            5.1 |           1.8 | Iris-virginica |

The column names of the dataset:
Index(['sepal length', 'sepal width', 'petal length', 'petal width', 'class'], dtype='object')

These print checks were conducted to ensure that the data was correctly imported and in the correct format.
The number of rows and columns in the dataset:
None

The number of missing values in the dataset:
|              |   0 |
|:-------------|----:|
| sepal length |   0 |
| sepal width  |   0 |
| petal length |   0 |
| petal width  |   0 |
| class        |   0 |

The number of duplicate rows in the dataset:
3

The data types of each column in the dataset:
|              | 0       |
|:-------------|:--------|
| sepal length | float64 |
| sepal width  | float64 |
| petal length | float64 |
| petal width  | float64 |
| class        | object  |

Missing values were checked for in the dataset, there were none. If there were missing values, the dataset would need to be cleaned and sorted further before any analysis could be conducted. There were no
missing values in this dataset, so further cleaning was unnecessary. From the information table, it can be seen that where one column has categorical (object) data (class column - also referred to as species
for this dataset)  and the four other columns (sepal length, sepal width, petal length, and petal width) are of the float type (float64) (continuous variables) with non-Null entries. That is, there are no 0 /
null~ entries in the dataset.

Duplicates were removed from the data using the drop_duplicates function.
The code used for this was: ```data = iris_df.drop_duplicates(subset=class.)```

Value counts for each of the species:
class
Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50
Name: count, dtype: int64

It can be seen that there are 50 samples for each of the three classes (species) of iris flower: Setosa, Versicolor, and Virginica.

Summary statistics for the whole dataset:
|       |   sepal length |   sepal width |   petal length |   petal width |
|:------|---------------:|--------------:|---------------:|--------------:|
| count |     150        |    150        |      150       |    150        |
| mean  |       5.84333  |      3.054    |        3.75867 |      1.19867  |
| std   |       0.828066 |      0.433594 |        1.76442 |      0.763161 |
| min   |       4.3      |      2        |        1       |      0.1      |
| 25%   |       5.1      |      2.8      |        1.6     |      0.3      |
| 50%   |       5.8      |      3        |        4.35    |      1.3      |
| 75%   |       6.4      |      3.3      |        5.1     |      1.8      |
| max   |       7.9      |      4.4      |        6.9     |      2.5      |

The summary statistics for the whole dataset shows that there are 150 samples in the dataset for each of the variables, the value displayed is the non-missing values, and thus it can be verified that the
dataset does not have any missing values. The Mean, Standard Deviation (std), Minimum (min) and Maximum (max), and the Lower, Median, and Higher Inter-quartile Ranges (25%, 50%, and 75%, respectively) values
are displayed for all four features (sepal length, sepal width, petal length, and petal width). The Mean was calculated by dividing the sum of all the values (per feature) by the number of values (150 in this
case). The mean for sepal length was 5.84 cm, sepal width was 3.05 cm, petal length was 3.76 cm, and for petal width was 1.20 cm. The Standard Deviation (std) is a measure of the spread of the data, that is, on
average, how much the values deviate from the mean. Sepal length had a mean of 5.84 cm with a std of 0.83, therefore the majority of values deviate by 0.83 (+/-) from the mean.  The mean for sepal width was
3.05 cm with a std of 0.43, the sepal widths deviated by +/- 0.43 cm from the mean. The feature, sepal width, had less variability (std = 0.43) compared to that of sepal length (std = 0.83).  For petal length,
the mean was 3.76 cm with a std of 1.76, thus most values for petal length deviated by 1.76 cm (+/-). Petal width had a mean of 1.20 cm with a std of 0.76, the width of petals deviated by +/- 0.76 cm.  The
measurement with the largest deviation from the mean is the petal length (std = 1.76), this suggests that petal lengths vary more widely across samples compared to the other features.

Outliers detected for each species:

### Outliers for Iris-setosa:

#### Column 'sepal length': No outliers detected

#### Column 'sepal width': No outliers detected

#### Column 'petal length': 4 outliers
|   sepal length |   sepal width |   petal length |   petal width | class       |
|---------------:|--------------:|---------------:|--------------:|:------------|
|            4.3 |           3   |            1.1 |           0.1 | Iris-setosa |
|            4.6 |           3.6 |            1   |           0.2 | Iris-setosa |
|            4.8 |           3.4 |            1.9 |           0.2 | Iris-setosa |
|            5.1 |           3.8 |            1.9 |           0.4 | Iris-setosa |

#### Column 'petal width': 2 outliers
|   sepal length |   sepal width |   petal length |   petal width | class       |
|---------------:|--------------:|---------------:|--------------:|:------------|
|            5.1 |           3.3 |            1.7 |           0.5 | Iris-setosa |
|            5   |           3.5 |            1.6 |           0.6 | Iris-setosa |

### Outliers for Iris-versicolor:

#### Column 'sepal length': No outliers detected

#### Column 'sepal width': No outliers detected

#### Column 'petal length': 1 outliers
|   sepal length |   sepal width |   petal length |   petal width | class           |
|---------------:|--------------:|---------------:|--------------:|:----------------|
|            5.1 |           2.5 |              3 |           1.1 | Iris-versicolor |

#### Column 'petal width': No outliers detected

### Outliers for Iris-virginica:

#### Column 'sepal length': 1 outliers
|   sepal length |   sepal width |   petal length |   petal width | class          |
|---------------:|--------------:|---------------:|--------------:|:---------------|
|            4.9 |           2.5 |            4.5 |           1.7 | Iris-virginica |

#### Column 'sepal width': 3 outliers
|   sepal length |   sepal width |   petal length |   petal width | class          |
|---------------:|--------------:|---------------:|--------------:|:---------------|
|            7.7 |           3.8 |            6.7 |           2.2 | Iris-virginica |
|            6   |           2.2 |            5   |           1.5 | Iris-virginica |
|            7.9 |           3.8 |            6.4 |           2   | Iris-virginica |

#### Column 'petal length': No outliers detected

#### Column 'petal width': No outliers detected

In the Summary Statistics for each species, the count shows that there are 50 samples in the dataset for each, the values displayed is the non-missing value, suggesting that there are no missing values present in the dataset.

## Individual Species Observations

### Iris Setosa

Setosa Statistics:
|       |   sepal length |   sepal width |   petal length |   petal width |
|:------|---------------:|--------------:|---------------:|--------------:|
| count |       50       |     50        |      50        |      50       |
| mean  |        5.006   |      3.418    |       1.464    |       0.244   |
| std   |        0.35249 |      0.381024 |       0.173511 |       0.10721 |
| min   |        4.3     |      2.3      |       1        |       0.1     |
| 25%   |        4.8     |      3.125    |       1.4      |       0.2     |
| 50%   |        5       |      3.4      |       1.5      |       0.2     |
| 75%   |        5.2     |      3.675    |       1.575    |       0.3     |
| max   |        5.8     |      4.4      |       1.9      |       0.6     |

The mean for sepal length was 5.006 cm, sepal width was 3.418 cm, petal length was 1.464 cm, and for petal width the mean was 0.244 cm. The mean was calculated by dividing the sum of all the values (per
feature) by the number of values (50 in this case, as it is done by species('class')). The standard deviation (std) is a measure of the spread of the data, that is, on average, how much the values deviate from
the mean. For sepal length the mean was 5.006 cm and the std was 0.35249, therefore most values deviated by 0.35249 cm (+/-) from the mean. The mean for sepal width was 3.418 cm and the std was 0.381024, so
most values deviated by +/- 0.381024 cm from the mean.Petal length had a mean of 1.464 cm and the std was 0.173511, therefore most values deviated by +/- 0.173511 cm from the mean. For petal width, a mean of
0.244 cm and the std of 0.10721 was calculated, therefore most values deviated by +/- 0.10721 cm from the mean. The median for sepal length was 5.0 cm, for sepal width it was 3.4 cm, for petal length it was 1.5
cm, and for petal width it was 0.2 cm. The median is the middle value of the data when sorted in increasing (ascending) order, therefore half of the values are above and half are below the median.The upper and
lower inter-quartile ranges (IQR) are the 25 % and 75 % values of the data, respectively. The IQR is a measure of the spread of the data, that is, how much the values deviate from the mean. For sepal length,
the IQR was 0.3 cm, for sepal width it was 0.2 cm, for petal length it was 0.4 cm, and for petal width it was 0.2 cm.

### Iris Versicolor

Versicolor Statistics:
|       |   sepal length |   sepal width |   petal length |   petal width |
|:------|---------------:|--------------:|---------------:|--------------:|
| count |      50        |     50        |      50        |     50        |
| mean  |       5.936    |      2.77     |       4.26     |      1.326    |
| std   |       0.516171 |      0.313798 |       0.469911 |      0.197753 |
| min   |       4.9      |      2        |       3        |      1        |
| 25%   |       5.6      |      2.525    |       4        |      1.2      |
| 50%   |       5.9      |      2.8      |       4.35     |      1.3      |
| 75%   |       6.3      |      3        |       4.6      |      1.5      |
| max   |       7        |      3.4      |       5.1      |      1.8      |

The mean for sepal length was 5.936 cm, sepal width was 2.77 cm, petal length was 4.26 cm, and for petal width the mean was 1.326 cm. The mean was calculated by dividing the sum of all the values (per feature)
by the number of values (50 in this case, as it is done by species('class')). For sepal length the mean was 5.936 cm and the std was 0.516171, therefore most values deviated by 0.516171 cm (+/-) from the mean.
The mean for sepal width was 2.77 cm and the std was 0.313798, so most values deviated by +/- 0.313798 cm from the mean. Petal length had a mean of 4.26 cm and the std was 0.469911, therefore most values
deviated by +/- 0.469911 cm from the mean. For petal width, a mean of 1.326 cm and the std of 0.197753 was calculated, therefore most values deviated by +/- 0.197753 cm from the mean. The median for sepal
length was 5.9 cm, for sepal width it was 2.8 cm, for petal length it was 4.6 cm, and for petal width it was 1.3 cm. The upper and lower inter-quartile ranges (IQR) for sepal length was 0.4 cm, for sepal width
it was 0.2 cm, for petal length it was 0.4 cm, and for petal width it was 0.2 cm.

### Iris Virginica

Virginica Statistics:
|       |   sepal length |   sepal width |   petal length |   petal width |
|:------|---------------:|--------------:|---------------:|--------------:|
| count |       50       |     50        |      50        |      50       |
| mean  |        6.588   |      2.974    |       5.552    |       2.026   |
| std   |        0.63588 |      0.322497 |       0.551895 |       0.27465 |
| min   |        4.9     |      2.2      |       4.5      |       1.4     |
| 25%   |        6.225   |      2.8      |       5.1      |       1.8     |
| 50%   |        6.5     |      3        |       5.55     |       2       |
| 75%   |        6.9     |      3.175    |       5.875    |       2.3     |
| max   |        7.9     |      3.8      |       6.9      |       2.5     |

The mean for sepal length was 6.588 cm, sepal width was 2.974 cm, petal length was 5.552 cm, and for petal width the mean was 2.026 cm. The mean was calculated by dividing the sum of all the values (per
feature) by the number of values (50 in this case, as it is done by species('class')). For sepal length the mean was 6.588 cm and the std was 0.63588, therefore most values deviated by 0.63588 cm (+/-) from the
mean. The mean for sepal width was 2.974 cm and the std was 0.322497, so most values deviated by +/- 0.322497 cm from the mean. Petal length had a mean of 5.552 cm and the std was 0.551895, therefore most
values deviated by +/- 0.551895 cm from the mean. For petal width, a mean of 2.026 cm and the std of 0.27465 was calculated, therefore most values deviated by +/- 0.27465 cm from the mean. The median for sepal
length was 6.5 cm, for sepal width it was 3.0 cm, for petal length it was 5.55 cm, and for petal width it was 2.0 cm. The upper and lower inter-quartile ranges (IQR) for sepal length was 0.5 cm, for sepal width
it was 0.2 cm, for petal length it was 0.5 cm, and for petal width it was 0.2 cm.
