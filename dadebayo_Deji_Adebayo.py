#!/usr/bin/env python
# coding: utf-8

# # Imported Libraries

# In[1]:


import math                         
import numpy as np                  
import pandas as pd               
import seaborn as sns              
import matplotlib.pyplot as plt   
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_theme(style = "darkgrid")
import warnings
warnings.filterwarnings('ignore')
plt.rcParams["figure.figsize"] = (15, 10)
import yfinance as yf
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn import metrics


# # Question 1.3

# In[2]:


################################################Download all 30 stocks on Dow Jones market of 1 year###############################################
data = yf.download("MMM AXP AMGN AAPL BA CAT CVX CSCO KO DIS DOW GS HD HON IBM INTC JNJ JPM MCD MRK MSFT NKE PG CRM TRV UNH VZ V WBA WMT", start="2020-01-01", end="2020-12-31")


# In[3]:


################################################View the dataframe###########################################################
data


# In[4]:


#################################Calculating the correlation Matrix################################################

adjCloseDataFrame = data['Adj Close'] #Selecting the Adjust close data into a dataframe

calculateDailyReturns = adjCloseDataFrame.pct_change() #Computing the daily returns

returnsWithOutNan = calculateDailyReturns.dropna() #Drop Nan columns

####################Correlation Matrix##############################

correlationMatrix = returnsWithOutNan.corr()

correlationMatrix


# In[5]:


################################Using the correlation Matrix for PCA and plotting the bar graph#####################

pcaModel = PCA()

fitPCAModel = pcaModel.fit(correlationMatrix)

storePCAComponent = fitPCAModel.components_

storeAdjCloseColumns = adjCloseDataFrame.columns

############################### Plot for PCA 1 #######################################

plt.bar(storeAdjCloseColumns, storePCAComponent[0])
plt.xticks(rotation=90)
plt.title("A Bar Graph to Show the Weight of the first PCA Stock")
plt.xlabel("Stock Names")
plt.ylabel("Weight")
plt.show()

############################### Plot for PCA 2 #######################################

plt.bar(storeAdjCloseColumns, storePCAComponent[1])
plt.xticks(rotation=90)
plt.title("A Bar Graph to Show the Weight of the second PCA Stock")
plt.xlabel("Stock Names")
plt.ylabel("Weight")
plt.show()


# In[6]:


# Is the first or second principal component similar to the market (equal weight on each stock)? Discuss why?


# In[7]:


#Download the Dow Jone Industrial average data
data_djia = yf.download('^DJI', start="2020-01-01", end="2020-12-31")

data_djia


# In[8]:


##################Compute the value of daily returns################
data_djia_adjColumn = data_djia['Adj Close']

data_djia['Returns'] = data_djia_adjColumn.pct_change()

data_djia


# In[9]:


#####################Calculate the weight for Principal Component 1##################################################
pc1 = storePCAComponent[0]

pc1Sum = sum(abs(pc1)) 

pc1WeightValues = abs(pc1)/(pc1Sum)

pc1WeightValues = np.array(pc1WeightValues)

returnsWeight = returnsWithOutNan * pc1WeightValues

returnsWeight['values'] = returnsWeight.sum(axis=1)

plt.plot(returnsWeight.index, returnsWeight['values'], label="Principal component 1")
plt.plot(data_djia.index, data_djia['Returns'], label = 'Dow Jones Instructional Average')
plt.ylabel("Weight of Stocks")
plt.xlabel("Dates")
plt.title("Comparing weight of PC 1 with the market")
plt.legend()
plt.show()

#####################Calculate the weight for Principal Component 2########################################
pc2 = storePCAComponent[1]

pc2Sum = sum(abs(pc2)) 

pc2WeightValues = abs(pc2)/(pc2Sum)

pc2WeightValues = np.array(pc2WeightValues)

returnsWeight2 = returnsWithOutNan * pc2WeightValues

returnsWeight2['values'] = returnsWeight2.sum(axis=1)

plt.plot(returnsWeight2.index, returnsWeight2['values'], label="Principal component 2")
plt.plot(data_djia.index, data_djia['Returns'], label = 'Dow Jones Instructional Average')
plt.ylabel("Weight of Stocks")
plt.xlabel("Dates")
plt.title("Comparing weight of PC 2 with the market")
plt.legend()
plt.show()


# # Question 1.4
# 
# 

# In[10]:


#Calculate the amount of variance explained by each principal component and make a ‘Scree’ plot. 

amountOfVarianceExplained = pcaModel.explained_variance_ratio_

returnCumSum = np.cumsum(amountOfVarianceExplained)

plt.plot(returnCumSum,  'o-', linewidth=2, color='blue')
plt.axhline(y=0.95, color='r', linestyle='--', label ="95% variance")
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.show()


# In[11]:


#How many principal components are required to explain 95% of the variance?
#Add up the values in the cummulative sum to know how many required to explain 95% of the variance

getNumberPcaRequired = next(x for x, val in enumerate(returnCumSum)if val > 0.95)+ 1

print(f"{getNumberPcaRequired}, is the number of principal components required to explain 95% of the variance")


# # Question 1.5

# In[12]:


# Investigate the scatter plot of the first two principal components and calculate the average
#of all 30 stocks. Based on Euclidean distances away from this average, identify the three
#most distant stocks. Can you explain why these stocks are unusual?

storeAdjCloseColumns = adjCloseDataFrame.columns
for x, storeAdjCloseColumns in enumerate(storeAdjCloseColumns):
     plt.annotate(storeAdjCloseColumns, xy=(storePCAComponent[0][x], storePCAComponent[1][x]))
plt.scatter(storePCAComponent[0], storePCAComponent[1], s=40, c='blue')
plt.title("A Scatter Plot of the First Two Principal Components")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

###############################Calculating the Average###############################
# changeToDataFrame = pd.DataFrame(storePCAComponent).T

averageOfStockPCA1 = np.mean(storePCAComponent[0])

averageOfStockPCA2 = np.mean(storePCAComponent[1])

print(f"{averageOfStockPCA1}, is the average of all stocks in PCA 1\n {averageOfStockPCA2} is the average of all stocks in PCA 2")

############################Euclidean Distance ########################## 
changeAverageStockList = np.array((averageOfStockPCA1, averageOfStockPCA2))

listDistance = []
for i in range (len(storePCAComponent)):
    distance = np.linalg.norm((changeAverageStockList)- np.array(storePCAComponent[0][i], storePCAComponent[1][i]))
    listDistance.append(distance)
# sortListDistance = sorted(listDistance, reverse=True)
# print(f"{listDistance}")



# In[13]:


#Dataframe to hold the Distance and Tickers 
storeAdjCloseColumns = adjCloseDataFrame.columns
df = pd.DataFrame(list(zip(storeAdjCloseColumns, listDistance)),columns =['Tickers', 'Distance'])

df


# In[14]:


#Largest values in the dataframe 
df.nlargest(3, ['Distance'])


# # Question 2.3

# In[15]:


#Use the correlation matrix from question (1.3) above to provide pairwise distances
# between the 30 stocks. Give the formula for this rescaled distance and provide an
# interpretation of small and large distances  (2(1 − ρij))1/2

pairwiseDistance =(2 *(1 - correlationMatrix))**(0.5)

pairwiseDistance


# # Question 2.4

# In[16]:


# Construct a horizontal dendrogram using the average linkage approach, carefully labelling the graphic with the names of the 30 stocks
#Dataframe to hold the Distance and Tickers 
storeAdjCloseColumns = adjCloseDataFrame.columns
getLinkage = linkage(pairwiseDistance, method='average')
dn = dendrogram(getLinkage, orientation = 'left', labels=storeAdjCloseColumns)
plt.show()


# # Question 2.5

# In[17]:


#Creating a cluster
storeAdjCloseColumns = adjCloseDataFrame.columns
getLinkage = linkage(pairwiseDistance, method='average')
dn = dendrogram(getLinkage, orientation = 'left', labels=storeAdjCloseColumns, color_threshold = 1.75)
plt.axvline(x=1.75, color='r', linestyle='--', label ="Cluster")
plt.show()


# In[18]:


#List of Clusters
for i in range(len(dn['ivl'])):
    print(f"{dn['ivl'][i]}, is in cluster {dn['leaves_color_list'][i]}")


# # Question 3.4

# In[19]:


##################################################Load Titanic Data into a DataFrame#####################################
titanicDataframe = pd.read_csv("titanic3.csv")

titanicDataframePredictor = titanicDataframe[['pclass', 'age', 'sex']]

#Get the mean of the  age column and fill  with the null
titanicDataframePredictor['age'] = titanicDataframePredictor['age'].fillna(titanicDataframePredictor['age'].mean())

titanicDataframePredictor['sex'] = titanicDataframePredictor["sex"].replace(["female", "male"], [0,1])

##################### Create and Fit the model for Random Forest###################

TreeList = list(range(1, 100, 10))

values = []

#iterate over the neighbour List
for k in TreeList:
    RandForestValueTree = RandomForestClassifier(n_estimators = k, random_state=0)
    RandForestValueTree.fit(titanicDataframePredictor, titanicDataframe["survived"])
    PredictionValue = RandForestValueTree.predict(titanicDataframePredictor)
    crossValidationValues = cross_validate(RandForestValueTree, titanicDataframePredictor, titanicDataframe["survived"], cv=5, scoring='balanced_accuracy')
    getMeanScore = np.mean(crossValidationValues['test_score'])
    values.append(getMeanScore)
    
plt.plot(TreeList, values, "ro-")
plt.text(81, max(values), "Optimal Point")
plt.ylabel("mean score value ")
plt.xlabel("Trees in the Forest")
plt.title("Random Forest")
plt.show()

###############################Getting the max in the list###################################
getMaxValuesInList = max(values)

print(f"The maximum value in the list is {getMaxValuesInList}, which is the optimal number of trees")


# # Question 3.5

# In[20]:


############################Random Forest##############################
RandForestClf = RandomForestClassifier(n_estimators=81, random_state=0)

RandForestClf.fit(titanicDataframePredictor, titanicDataframe["survived"])

#Logistic Regression

lRegression = linear_model.LogisticRegression()

lRegression.fit(titanicDataframePredictor, titanicDataframe["survived"])

#KNN
classifierValue = KNeighborsClassifier(n_neighbors= 5, metric="euclidean")

classifierValue.fit(titanicDataframePredictor, titanicDataframe["survived"])

#Tree Classification
TreeModel = DecisionTreeClassifier(random_state=0, ccp_alpha=0.010)

TreeModel.fit(titanicDataframePredictor, titanicDataframe["survived"])


# In[21]:


#Plot the metrics
holdMetrics = metrics.plot_roc_curve(RandForestClf, titanicDataframePredictor, titanicDataframe["survived"])
metrics.plot_roc_curve(lRegression, titanicDataframePredictor, titanicDataframe["survived"], ax=holdMetrics.ax_) 
metrics.plot_roc_curve(classifierValue, titanicDataframePredictor, titanicDataframe["survived"], ax=holdMetrics.ax_) 
metrics.plot_roc_curve(TreeModel, titanicDataframePredictor, titanicDataframe["survived"], ax=holdMetrics.ax_) 
plt.show()


# # Question 4.2

# In[22]:


##########################################Random Forest for Red Wine########################################### 
redWineDataFrame = pd.read_csv("winequality-red.csv")

redWineDataFrame[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides','free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH','sulphates', 'alcohol','quality']] = redWineDataFrame['fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"'].str.split(';', expand=True)

removeDefaultColumns = ['fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"']

redWineDataFrame.drop(removeDefaultColumns, axis = 1, inplace=True)

redWineDataFrame = redWineDataFrame.astype(float)

redWineDataFrameCorr = redWineDataFrame.loc[:, redWineDataFrame.columns != 'quality']

###########################Building Random Forest#########################################

TreeListRedWine = list(range(2, 100, 10))

valuesRedWine = []

#iterate over the List
for k in TreeListRedWine:
    RandForestValueLeaf = RandomForestClassifier(max_leaf_nodes = k, random_state=0)
    RandForestValueLeaf.fit(redWineDataFrameCorr, redWineDataFrame["quality"])
    PredictionValue = RandForestValueLeaf.predict(redWineDataFrameCorr)
    crossValidationValues = cross_validate(RandForestValueLeaf, redWineDataFrameCorr, redWineDataFrame["quality"], cv=5, scoring='balanced_accuracy')
    getMeanScore = np.mean(crossValidationValues['test_score'])
    valuesRedWine.append(getMeanScore)

plt.plot(TreeListRedWine, valuesRedWine, "go-")
plt.text(82, max(valuesRedWine), "Optimal Point")
plt.ylabel("mean values of scores")
plt.xlabel("Leaf in the Forest")
plt.title("Graph of Random Forest for RedWine Optimal Leaf")
plt.show()

###############################Getting the max in the list###################################
getMaxValuesInListRedWine = max(valuesRedWine)

print(f"The maximum value in the list is {getMaxValuesInListRedWine}, which is the optimal number of trees")


# # Question 4.3

# In[23]:


##########################################Random Forest for Red Wine########################################### 
redWineDataFrame = pd.read_csv("winequality-red.csv")

redWineDataFrame[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides','free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH','sulphates', 'alcohol','quality']] = redWineDataFrame['fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"'].str.split(';', expand=True)

removeDefaultColumns = ['fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"']

redWineDataFrame.drop(removeDefaultColumns, axis = 1, inplace=True)

redWineDataFrame = redWineDataFrame.astype(float)

redWineDataFrameCorr = redWineDataFrame.loc[:, redWineDataFrame.columns != 'quality']


###########################Building Random Forest#########################################

TreeListRedWine = list(range(1, 100, 10))

valuesRedWine = []

#iterate over the neighbour List
for k in TreeListRedWine:
    RandForestValue = RandomForestClassifier(n_estimators = k, random_state=0)
    RandForestValue.fit(redWineDataFrameCorr, redWineDataFrame["quality"])
    PredictionValue = RandForestValue.predict(redWineDataFrameCorr)
    crossValidationValues = cross_validate(RandForestValue, redWineDataFrameCorr, redWineDataFrame["quality"], cv=5, scoring='balanced_accuracy')
    getMeanScore = np.mean(crossValidationValues['test_score'])
    valuesRedWine.append(getMeanScore)

plt.plot(TreeListRedWine, valuesRedWine, "go-")
plt.text(50, max(valuesRedWine), "Optimal Point")
plt.ylabel("mean score value")
plt.xlabel("Trees in the Forest")
plt.title("Random Forest for optimal tree")
plt.show()

###############################Getting the max in the list###################################
getMaxValuesInListRedWine = max(valuesRedWine)

print(f"The maximum value in the list is {getMaxValuesInListRedWine}, which is the optimal number of trees")


# # Question 4.4 

# In[24]:


############################Lasso#####################################################
RedCorr = redWineDataFrameCorr.corrwith(redWineDataFrame["quality"])

storeCols = redWineDataFrameCorr.columns

lassoModel = linear_model.Lasso(alpha = 0.0001, random_state = 0)

redWhineFit = lassoModel.fit(redWineDataFrameCorr, redWineDataFrame["quality"])

coefRedWine = redWhineFit.coef_

storeFeatureImportance = RandForestValue.feature_importances_
plt.bar(storeCols, storeFeatureImportance, color='green')
plt.xticks(rotation=90)
plt.ylabel("values")
plt.xlabel("Red WIine Features")
plt.title("Bar graph of Feature importance using random forest model")
plt.show()


# In[25]:


plt.bar(storeCols, RedCorr)
plt.xticks(rotation=90)
plt.ylabel("values")
plt.xlabel("Red WIine Features")
plt.title("Bar graph of Correlation on the quality of red wine columns")
plt.show()


# In[26]:


plt.bar(storeCols, coefRedWine, color='brown')
plt.xticks(rotation=90)
plt.ylabel("values")
plt.xlabel("Red WIine Features")
plt.title("Bar graph of Correlation of Red Wine using Lasso")
plt.show()


# # Question 4.5

# In[27]:


###########################Linear Regression Performance########################
#Linear regression
regModel = linear_model.LinearRegression()

regModel.fit(redWineDataFrameCorr, redWineDataFrame["quality"])
# res = regModel.score(redWineDataFrameCorr, redWineDataFrame["quality"])

res = cross_validate(regModel, redWineDataFrameCorr, redWineDataFrame["quality"], cv=5, scoring='neg_mean_absolute_error')

finalRes = np.mean(abs(res['test_score']))

print(f"The mean square error (Linear Regression) {finalRes}")


# In[28]:


###############################################KNN##########################################
storeRegressor = KNeighborsRegressor()

storeRegressor.fit(redWineDataFrameCorr, redWineDataFrame["quality"])

firstCrossValidation = cross_validate(storeRegressor, redWineDataFrameCorr, redWineDataFrame["quality"], cv=5, scoring = 'neg_mean_absolute_error')

firstCrossValidation = np.mean(abs(firstCrossValidation['test_score']))

print(f"The mean square error(KNN): {firstCrossValidation}")


# In[29]:


##############################Random Forest###############################################
RandForestClfRedWine = RandomForestClassifier(n_estimators=50, max_leaf_nodes= 82, random_state=0)

RandForestClfRedWine.fit(redWineDataFrameCorr, redWineDataFrame["quality"])

RandForestPerformance = cross_validate(RandForestClf, redWineDataFrameCorr, redWineDataFrame["quality"], cv=5, scoring = 'neg_mean_absolute_error')

getMeanScoreRedWine = np.mean(abs(RandForestPerformance['test_score']))

print(f"The mean square error(RF): {getMeanScoreRedWine}")


# In[ ]:





# In[ ]:




