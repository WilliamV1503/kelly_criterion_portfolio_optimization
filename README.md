# kelly_criterion_portfolio_optimization
Application of Kelly Criterion for assigning weights to stocks in a portfolio. 

Portfolio 1: Daily Rolling Kelly - Use the previous day's covariance matrix with n-day lookback to calculate Kelly fractions for the next day

Portfolio 2: Forecasated Daily Rolling Kelly - Use k historical n-day-lookback covariance matrices to forecast the next day's covariance matrix to calculate Kelly fractions
* The sum of squared errors of all entries in each day's forecasted Cholesky factor is also plotted over the entire trading period. 

Portfolio 3: Monthly Rolling Kelly - Use the previous month's covariance matrix with n-month lookback to calculate Kelly fractions for the next month (effective at start of month)

Different metrics and return plots are generated for each portfolio. 


To Do: 


Optimize time series generation and model selection for each Cholesky factor entry. 

Forecast covariance matrices for Monthly Rolling Kelly

