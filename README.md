# Simon Lipson Projects
Here you will find projects of mine mainly focused on machine learning.
## AFL Brownlow Medal Predictor
![](https://cdn4.theroar.com.au/wp-content/uploads/2012/09/Brownlow-Medal-415x285.jpg)

Australian Rules Football is a fantastic game to watch and a brilliant sport to predict. Each year, one player is awarded the Brownlow Medal (the MVP of the season). After each game, the on-field referees decide which player should receive one-vote, two-votes and three-votes. At the end of the season, all of those votes are tallied and the player with the most wins the medal. 
This of course draws a lot of betting and speculation to determine who will win the medal. And thus, the problem: 

**Can an algorithm help predict the winner of the Brownlow Medal?**

To help me do so I attained data from the 2003-2019 AFL seasons and trained a model to predict the amount of votes each player will receive in each of those seasons. The player that was predicted to have the highest amount of votes should theoretically be the winner of the medal. 

[The code can be found here](https://github.com/simonlipson/simon_lipson_projects.github.io/blob/master/AFL%20Brownlow%20Predictor.ipynb)

The project followed the following steps:
1. Collect the data
- [kaggle.com](https://www.kaggle.com/stoney71/aflstats)
- [afltables.com](www.afltables.com)
2. Clean and normalize the data
3. Feature engineering through aggregation
4. Train, test splitting of the data by years
5. Evaluating regression models
6. Tuning hyperparameters of the model
7. Fitting the model with training data
8. Generating predictions with test data
9. Visualizing the outcomes in appropriate way

Follow all the above steps in the [notebook](https://github.com/simonlipson/simon_lipson_projects.github.io/blob/master/AFL%20Brownlow%20Predictor.ipynb). 
See below for the highlights.

### Evaluating Different Regression Algorithms

I used a k-fold cross validation to evaluate the accuracy of several different regression models. This validation optimized for r-squared which is a standard metric for regression algorithm evaluation. See the results below:

![alt text](images/cv_graph.png)
