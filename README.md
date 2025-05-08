# Project for the Programming and Scripting Module

## What is it all about?

The iris dataset has been a tool used for machine learning and for learning the basics of data analytics and programming.
This repository will be used to save the iris dataset as a csv and then also run analysis in a separate python file.


## Problem statement - DELETE
This project concerns the well-known Fisher’s Iris data set [3]. You must research the data set
and write documentation and code (in Python [1]) to investigate it. An online search for
information on the data set will convince you that many people have investigated it
previously. You are expected to be able to break this project into several smaller tasks that
are easier to solve, and to plug these together after they have been completed.
You might do that for this project as follows:

1. Research the data set online and write a summary about it in your README.
2. Download the data set and add it to your repository.
3. Write a program called analysis.py that:
    1. Outputs a summary of each variable to a single text file,
    2. Saves a histogram of each variable to png files, and
    3. Outputs a scatter plot of each pair of variables.
    4. Performs any other analysis you think is appropriate.

## Basic Data Checks

The iris dataset contains 150 samples of 3 species, there are 50 samples for each of the species. Each sample has 4 measurements taken (other than species/class), these are: petal length, petal width, sepal length, and sepal width.

## Explain what the code does for the important tests

## Background

The iris dataset (originally sourced by Anderson in 1935) is a well examined and thoroughly evaluated dataset of three species of iris flower (Setosa, Versicolor, Virginica) totalling 150 samples measuring four key features (sepal length/width, petal length/width) that aid in differentiation between the species. Not only highlighting the importance of field expert knowledge when designing a research project and its analysis, but also highlighting that there can be overlap between different species within the same genus (highlighted by the repeated overlap for Versicolor and Virginica for some features). In 1936, Fisher used the dataset as an example of linear discriminant analysis, used for finding a linear combination of features that chan characterise or separate classes or objects (specifically two or more). Throughout the decades, the iris dataset has been used thoroughly as an example dataset for machine learning and algorithm development and is stored as an open access dataset in the UCI Machine Learning Repository.

## Overview of project

This project focused on using the data in a number of different statistical tests to see what patterns were visible in the data, whether there were indeed differences between the species, and what pairs of features had influence over the other. #

Initial exploratory data analysis described the shape of the dataset (150 samples 5 variables). A check on first and last five rows of the dataset was conduced, and the presence of missing values was checked (none were found). The summary statistics were initially conducted on the dataset as a whole, before being split by species for a more granular analysis. For the overall summary statistics it was found that petal length had the largest standard deviation (difference from mean value) of 1.76 respectively, suggesting that they varied more widely across samples compared to sepal features, indicating that maybe this would be useful feature to investigate further to see where these differences were.

Boxplots were plotted for each of the four features with the samples separated by species

## Conclusion

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