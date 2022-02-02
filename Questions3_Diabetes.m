dataFrame = xlsread("titanic3.csv"); %Load data into a dataframe

independentVariables = dataFrame(:,[2 5:6]); % Select needed independent data from the data frame
dependentVariables = dataFrame(:,[2]); % Select needed dependent data from the data frame

correlation_Coeff = corrcoef(independentVariables) %Store Correlation Coefficient

figure
heatMap = heatmap(correlation_Coeff)
heatMap.Title = "Heat Map of the Matrix Correlation";
heatMap.XLabel = "independent_Variable_X";
heatMap.YLabel = "independent_Variable_Y";
figure
computemodel = fitlm(independentVariables,dependentVariables);
disp(computemodel)
% stepwise regression
stepwise_model = stepwise(independentVariables, dependentVariables)