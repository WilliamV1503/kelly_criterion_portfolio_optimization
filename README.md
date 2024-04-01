# kelly_criterion_portfolio_optimization
Application of Kelly Criterion for assigning weights to stocks in a portfolio. 

The Jupyter Notebook includes functions and the class used to generate time series data and SVR models used to forecast next-day entries of lower triangular Cholesky factor of trailing 252-day covariance matrix. It also quantifies the performance of three long-only, non-leveraged portfolios as described:

Portfolio 1: Use the previous day's covariance matrix to calculate Kelly fractions for the next day (not using the entire balance for the next day)

Portfolio 2: Use the previous day's covariance matrix to calculate Kelly fractions for the next day (use the entire balance for the next day)

Portfolio 2: Forecast the next day's covariance matrix to calculate Kelly fractions for the next day (use the entire balance for the next day)

Different metrics and return plots are generated for each portfolio. 

The sum of squared errors of all entries in each day's forecasted Cholesky factor is also plotted over the entire trading period. 


To Do: 

Adjust objective function so that weights aren't near zero when the fractions aren't forced to sum up to 1 (or consider adding more stocks to portfolio)

Optimize time series generation and model selection for each Cholesky factor entry. 

Obtain better metric for the risk-free rate

