# Choosing E-Commerce Platform Development with Linear Regression

One day, I just got some contract work with an E-Commerce company based in New York City that sells clothing online but they also have in-store style and clothing advice sessions. Customers come in to the store, have sessions/meetings with a personal stylist, then they can go home and order either on a mobile app or website for the clothes they want.

The company is trying to decide whether to focus their efforts on their mobile app experience or their website. They've hired me on contract to help them figure it out! So, I decide to use Machine Learning modeling with Linear Regression method to get insight for this problem.

<b> Source dataset: Fake Dataset for Project Practice in Purwadhika School </b>

## Imports Libraries
Import the necessary libraries. 


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
```

## Get the Data

I will work with the E-Commerce Customers csv file from the company. It has Customer info, such as Email, Address, and their color Avatar. Then it also has numerical value columns. 

* Avg. Session Length: Average session of in-store style advice sessions.
* Time on App: Average time spent on App in minutes
* Time on Website: Average time spent on Website in minutes
* Length of Membership: How many years the customer has been a member. 

> **All this info are fake, so there is no confidential issue.**



```python
customers = pd.read_csv('Ecommerce Customers')
```


```python
customers.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Email</th>
      <th>Address</th>
      <th>Avatar</th>
      <th>Avg. Session Length</th>
      <th>Time on App</th>
      <th>Time on Website</th>
      <th>Length of Membership</th>
      <th>Yearly Amount Spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>mstephenson@fernandez.com</td>
      <td>835 Frank Tunnel\nWrightmouth, MI 82180-9605</td>
      <td>Violet</td>
      <td>34.497268</td>
      <td>12.655651</td>
      <td>39.577668</td>
      <td>4.082621</td>
      <td>587.951054</td>
    </tr>
    <tr>
      <td>1</td>
      <td>hduke@hotmail.com</td>
      <td>4547 Archer Common\nDiazchester, CA 06566-8576</td>
      <td>DarkGreen</td>
      <td>31.926272</td>
      <td>11.109461</td>
      <td>37.268959</td>
      <td>2.664034</td>
      <td>392.204933</td>
    </tr>
    <tr>
      <td>2</td>
      <td>pallen@yahoo.com</td>
      <td>24645 Valerie Unions Suite 582\nCobbborough, D...</td>
      <td>Bisque</td>
      <td>33.000915</td>
      <td>11.330278</td>
      <td>37.110597</td>
      <td>4.104543</td>
      <td>487.547505</td>
    </tr>
    <tr>
      <td>3</td>
      <td>riverarebecca@gmail.com</td>
      <td>1414 David Throughway\nPort Jason, OH 22070-1220</td>
      <td>SaddleBrown</td>
      <td>34.305557</td>
      <td>13.717514</td>
      <td>36.721283</td>
      <td>3.120179</td>
      <td>581.852344</td>
    </tr>
    <tr>
      <td>4</td>
      <td>mstephens@davidson-herman.com</td>
      <td>14023 Rodriguez Passage\nPort Jacobville, PR 3...</td>
      <td>MediumAquaMarine</td>
      <td>33.330673</td>
      <td>12.795189</td>
      <td>37.536653</td>
      <td>4.446308</td>
      <td>599.406092</td>
    </tr>
  </tbody>
</table>
</div>



> **After get the data, I checked the contain of dataset. It seems clean enough, so we can start the EDA.**


```python
customers.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Avg. Session Length</th>
      <th>Time on App</th>
      <th>Time on Website</th>
      <th>Length of Membership</th>
      <th>Yearly Amount Spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>33.053194</td>
      <td>12.052488</td>
      <td>37.060445</td>
      <td>3.533462</td>
      <td>499.314038</td>
    </tr>
    <tr>
      <td>std</td>
      <td>0.992563</td>
      <td>0.994216</td>
      <td>1.010489</td>
      <td>0.999278</td>
      <td>79.314782</td>
    </tr>
    <tr>
      <td>min</td>
      <td>29.532429</td>
      <td>8.508152</td>
      <td>33.913847</td>
      <td>0.269901</td>
      <td>256.670582</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>32.341822</td>
      <td>11.388153</td>
      <td>36.349257</td>
      <td>2.930450</td>
      <td>445.038277</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>33.082008</td>
      <td>11.983231</td>
      <td>37.069367</td>
      <td>3.533975</td>
      <td>498.887875</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>33.711985</td>
      <td>12.753850</td>
      <td>37.716432</td>
      <td>4.126502</td>
      <td>549.313828</td>
    </tr>
    <tr>
      <td>max</td>
      <td>36.139662</td>
      <td>15.126994</td>
      <td>40.005182</td>
      <td>6.922689</td>
      <td>765.518462</td>
    </tr>
  </tbody>
</table>
</div>




```python
customers.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 500 entries, 0 to 499
    Data columns (total 8 columns):
    Email                   500 non-null object
    Address                 500 non-null object
    Avatar                  500 non-null object
    Avg. Session Length     500 non-null float64
    Time on App             500 non-null float64
    Time on Website         500 non-null float64
    Length of Membership    500 non-null float64
    Yearly Amount Spent     500 non-null float64
    dtypes: float64(5), object(3)
    memory usage: 31.4+ KB


> **After put the csv data to the dataframe "customers", I checked the data cleanliness. I didn't found any anomali (like a null cell), so we can continue to eksplore and analyze the data.**

## Exploratory Data Analysis

Let's explore the data!


```python
from scipy.stats import pearsonr #I use this additional libraries to show the correlation value on the plot.
sns.set(style="darkgrid", color_codes=True)
```

> **I will use Seaborn to create a jointplot to compare the <font color=royalblue>Time on Website</font>  and <font color=royalblue>Yearly Amount Spent</font> columns.** 


```python
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers).annotate(pearsonr)
```




    <seaborn.axisgrid.JointGrid at 0x120cfba90>




![png](output_13_1.png)


> **Then, compare the <font color=royalblue>Time on App</font> and <font color=royalblue>Yearly Amount Spent</font> columns.** 


```python
sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers).annotate(pearsonr)
```




    <seaborn.axisgrid.JointGrid at 0x11e13a850>




![png](output_15_1.png)


> **Now, I use Seaborn Jointplot to create a 2D hex bin plot comparing <font color=royalblue>Time on App</font> and <font color=royalblue>Length of Membership</font>.**


```python
sns.jointplot(x = 'Time on App', y='Length of Membership', data=customers, kind='hex').annotate(pearsonr)
```




    <seaborn.axisgrid.JointGrid at 0x11e30c390>




![png](output_17_1.png)


> **Let's explore these types of relationships across the entire data set. Use [pairplot](https://stanford.edu/~mwaskom/software/seaborn/tutorial/axis_grids.html#plotting-pairwise-relationships-with-pairgrid-and-pairplot) to recreate the plot below.**


```python
sns.pairplot(customers).annotate(pearsonr)
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-10-c1fde6f631b3> in <module>
    ----> 1 sns.pairplot(customers).annotate(pearsonr)
    

    AttributeError: 'PairGrid' object has no attribute 'annotate'



![png](output_19_1.png)


> <b>Based of this plot we can look that <font color=royalblue>Length of Membership</font> column is the most correlated feature with <font color=royalblue>Yearly Amount Spent</font>. It makes sense because the longer you are become a member the bigger your possibility to spent more money on it. We can see that on the correlation value table also.</b>


```python
customers.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Avg. Session Length</th>
      <th>Time on App</th>
      <th>Time on Website</th>
      <th>Length of Membership</th>
      <th>Yearly Amount Spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Avg. Session Length</td>
      <td>1.000000</td>
      <td>-0.027826</td>
      <td>-0.034987</td>
      <td>0.060247</td>
      <td>0.355088</td>
    </tr>
    <tr>
      <td>Time on App</td>
      <td>-0.027826</td>
      <td>1.000000</td>
      <td>0.082388</td>
      <td>0.029143</td>
      <td>0.499328</td>
    </tr>
    <tr>
      <td>Time on Website</td>
      <td>-0.034987</td>
      <td>0.082388</td>
      <td>1.000000</td>
      <td>-0.047582</td>
      <td>-0.002641</td>
    </tr>
    <tr>
      <td>Length of Membership</td>
      <td>0.060247</td>
      <td>0.029143</td>
      <td>-0.047582</td>
      <td>1.000000</td>
      <td>0.809084</td>
    </tr>
    <tr>
      <td>Yearly Amount Spent</td>
      <td>0.355088</td>
      <td>0.499328</td>
      <td>-0.002641</td>
      <td>0.809084</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



> **Create a linear model plot (using seaborn's lmplot) of  <font color=royalblue>Yearly Amount Spent</font> vs. <font color=royalblue>Length of Membership</font>.**


```python
sns.jointplot(x='Length of Membership', y='Yearly Amount Spent', data=customers, kind='reg').annotate(pearsonr)
```




    <seaborn.axisgrid.JointGrid at 0x11f46f9d0>




![png](output_23_1.png)


## Training and Testing Data

Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets.<br>


```python
customers.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Email</th>
      <th>Address</th>
      <th>Avatar</th>
      <th>Avg. Session Length</th>
      <th>Time on App</th>
      <th>Time on Website</th>
      <th>Length of Membership</th>
      <th>Yearly Amount Spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>mstephenson@fernandez.com</td>
      <td>835 Frank Tunnel\nWrightmouth, MI 82180-9605</td>
      <td>Violet</td>
      <td>34.497268</td>
      <td>12.655651</td>
      <td>39.577668</td>
      <td>4.082621</td>
      <td>587.951054</td>
    </tr>
    <tr>
      <td>1</td>
      <td>hduke@hotmail.com</td>
      <td>4547 Archer Common\nDiazchester, CA 06566-8576</td>
      <td>DarkGreen</td>
      <td>31.926272</td>
      <td>11.109461</td>
      <td>37.268959</td>
      <td>2.664034</td>
      <td>392.204933</td>
    </tr>
    <tr>
      <td>2</td>
      <td>pallen@yahoo.com</td>
      <td>24645 Valerie Unions Suite 582\nCobbborough, D...</td>
      <td>Bisque</td>
      <td>33.000915</td>
      <td>11.330278</td>
      <td>37.110597</td>
      <td>4.104543</td>
      <td>487.547505</td>
    </tr>
    <tr>
      <td>3</td>
      <td>riverarebecca@gmail.com</td>
      <td>1414 David Throughway\nPort Jason, OH 22070-1220</td>
      <td>SaddleBrown</td>
      <td>34.305557</td>
      <td>13.717514</td>
      <td>36.721283</td>
      <td>3.120179</td>
      <td>581.852344</td>
    </tr>
    <tr>
      <td>4</td>
      <td>mstephens@davidson-herman.com</td>
      <td>14023 Rodriguez Passage\nPort Jacobville, PR 3...</td>
      <td>MediumAquaMarine</td>
      <td>33.330673</td>
      <td>12.795189</td>
      <td>37.536653</td>
      <td>4.446308</td>
      <td>599.406092</td>
    </tr>
  </tbody>
</table>
</div>



> **I set a variable "X" equal to the numerical features of the customers and a variable "Y" equal to the <font color=royalblue>Yearly Amount Spent</font> column.**


```python
X = customers.select_dtypes(exclude='object').drop('Yearly Amount Spent', axis=1)
Y = customers['Yearly Amount Spent']
```

> **I use model_selection.train_test_split from sklearn library to split the data into training and testing sets. <br> I set test_size=0.3 and random_state=101. It means I split the existing dataset into 70% for data training, and 30% for data test**


```python
from sklearn.model_selection import train_test_split 
```


```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
```

## Training the Model

Now its time to train our model on our training data!

> **I need to import LinearRegression from sklearn.linear_model library.**


```python
from sklearn.linear_model import LinearRegression
```

> **Then, create an instance of a LinearRegression() model named lm.**


```python
lm = LinearRegression()
```

> **Train/fit lm on the training data.**


```python
lm.fit(X_train,Y_train)
```




    LinearRegression()



> **Print out the coefficients of the model**


```python
lm.coef_
```




    array([25.98154972, 38.59015875,  0.19040528, 61.27909654])



## Predicting Test Data
Now that we have fit our model, let's evaluate its performance by predicting off the test values!

> **Use lm.predict() to predict off the X_test set of the data.**


```python
prediction = lm.predict(X_test)
```


```python
prediction
```




    array([456.44186104, 402.72005312, 409.2531539 , 591.4310343 ,
           590.01437275, 548.82396607, 577.59737969, 715.44428115,
           473.7893446 , 545.9211364 , 337.8580314 , 500.38506697,
           552.93478041, 409.6038964 , 765.52590754, 545.83973731,
           693.25969124, 507.32416226, 573.10533175, 573.2076631 ,
           397.44989709, 555.0985107 , 458.19868141, 482.66899911,
           559.2655959 , 413.00946082, 532.25727408, 377.65464817,
           535.0209653 , 447.80070905, 595.54339577, 667.14347072,
           511.96042791, 573.30433971, 505.02260887, 565.30254655,
           460.38785393, 449.74727868, 422.87193429, 456.55615271,
           598.10493696, 449.64517443, 615.34948995, 511.88078685,
           504.37568058, 515.95249276, 568.64597718, 551.61444684,
           356.5552241 , 464.9759817 , 481.66007708, 534.2220025 ,
           256.28674001, 505.30810714, 520.01844434, 315.0298707 ,
           501.98080155, 387.03842642, 472.97419543, 432.8704675 ,
           539.79082198, 590.03070739, 752.86997652, 558.27858232,
           523.71988382, 431.77690078, 425.38411902, 518.75571466,
           641.9667215 , 481.84855126, 549.69830187, 380.93738919,
           555.18178277, 403.43054276, 472.52458887, 501.82927633,
           473.5561656 , 456.76720365, 554.74980563, 702.96835044,
           534.68884588, 619.18843136, 500.11974127, 559.43899225,
           574.8730604 , 505.09183544, 529.9537559 , 479.20749452,
           424.78407899, 452.20986599, 525.74178343, 556.60674724,
           425.7142882 , 588.8473985 , 490.77053065, 562.56866231,
           495.75782933, 445.17937217, 456.64011682, 537.98437395,
           367.06451757, 421.12767301, 551.59651363, 528.26019754,
           493.47639211, 495.28105313, 519.81827269, 461.15666582,
           528.8711677 , 442.89818166, 543.20201646, 350.07871481,
           401.49148567, 606.87291134, 577.04816561, 524.50431281,
           554.11225704, 507.93347015, 505.35674292, 371.65146821,
           342.37232987, 634.43998975, 523.46931378, 532.7831345 ,
           574.59948331, 435.57455636, 599.92586678, 487.24017405,
           457.66383406, 425.25959495, 331.81731213, 443.70458331,
           563.47279005, 466.14764208, 463.51837671, 381.29445432,
           411.88795623, 473.48087683, 573.31745784, 417.55430913,
           543.50149858, 547.81091537, 547.62977348, 450.99057409,
           561.50896321, 478.30076589, 484.41029555, 457.59099941,
           411.52657592, 375.47900638])



> **Create a scatterplot of the real test values versus the predicted values.**


```python
sns.scatterplot(Y_test, prediction)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11ea5ab10>




![png](output_43_1.png)


> **It seems that our Model has a good prediction.**

## Evaluating the Model

Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2).

> **Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.** <br>
Reference: [studytonight.com](https://www.studytonight.com/post/what-is-mean-squared-error-mean-absolute-error-root-mean-squared-error-and-r-squared)


```python
from sklearn import metrics 
```


```python
print('MAE:', metrics.mean_absolute_error(Y_test, prediction))
print('MSE:', metrics.mean_squared_error(Y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, prediction)))
print('R2 Score:', metrics.r2_score(Y_test,prediction))
```

    MAE: 7.2281486534308295
    MSE: 79.8130516509743
    RMSE: 8.933815066978626
    R2 Score: 0.9890046246741234


## Residuals

We should have gotten a very good model with a good fit. Let's quickly explore the residuals to make sure everything was okay with our data. 

> **I plotted a histogram of the residuals, it looks normally distributed.**


```python

```


![png](output_49_0.png)


## Conclusion
We still want to figure out the answer to the original question, do we focus our effort on mobile app or website development? Or maybe that doesn't even really matter, and Membership Time is what is really important.  Let's see if we can interpret the coefficients at all to get an idea.

>**Let's see the Coefficient dataframe below.**


```python

```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coeffecient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Avg. Session Length</th>
      <td>25.981550</td>
    </tr>
    <tr>
      <th>Time on App</th>
      <td>38.590159</td>
    </tr>
    <tr>
      <th>Time on Website</th>
      <td>0.190405</td>
    </tr>
    <tr>
      <th>Length of Membership</th>
      <td>61.279097</td>
    </tr>
  </tbody>
</table>
</div>



**And this is what I suggest to the E-Commerce company:**
> **We can see that Lengh of Membership has a top coefficient value. It means that is the most important variable to get the high customer's <font color=royalblue>Yearly Amount Spent</font>. Let's see another coefficient value. <font color=royalblue>Time on App</font> has the bigger value than <font color=royalblue>Time on Website</font>. <font color=orangered>It indicates us to focus our effort on Mobile App</font> rather than Website Development. <br>
We should make a good UI/UX on the Mobile App to make customer feels great during their time on the app. It will lead to the longer Length of Membership, and will be converted to the better <font color=royalblue>Yearly Amount Spent</font> by the customer.**

## Thank You

See you in another Data Exploration.

**BR,<br>
Erwindra Rusli<br>
Data Scientist Student in Purwadhika School**
