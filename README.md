# DataNormalisationInPython

In this tutorial blog, I'm going to cover how to normalize data in python easily. I'll also show actual implementation by coding for the same. The jupyter Notebook for this tutorial can be viewed on [GitHub](https://github.com/Yadavbalbir/DataNormalisationInPython)
## What is Normalization?
Normalization of data is basically scaling of data. Which is commonly scaled between 0 and 1.

### Why do we need to Normalize data?
It is observed that machine learning algorithms tend to perform better and converge faster when the data are of a smaller scale.
So, it is always recommended and it is common practice to normalize your data.
Once you have normalized your data then you can build a model. 

Normalization decreases the **sensitivity** to the scale of the features resulting in **better results** after training.

####Formula for normalization is given below 
`Xnorm = (X-Xmin)/(Xmax-Xmin)`

Just subtract the data point with min of all the data point and then divide the result with the difference with the maximum and minimum of all data points.

### Normalization of data using Sklearn`
`from sklearn import preprocessing
import numpy as np`

`x=np.linspace(10, 100, 10)
x`
Ourput: `array([ 10.,  20.,  30.,  40.,  50.,  60.,  70.,  80.,  90., 100.])`

**Note:** linspace(a,b,size) is used to array which starts with a and ends on b with total number of elements equal to size.

Now, Let's Normalize the array x that we have created just above.

`normalized_x=preprocessing.normalize([x])
normalized_x`

**Output:** `array([[0.05096472, 0.10192944, 0.15289416, 0.20385888, 0.2548236 ,
        0.30578831, 0.35675303, 0.40771775, 0.45868247, 0.50964719]])`

#### Boom!! Array has been normalized
You can see that every element in the normalized array has been rescaled between zero to one.


## Another way to Normalize Data in python by using MinMaxScaler() 
MinMaxScaler() is also provided by sklearn. It is also good to know here what MinMaxScaler gives us an option to rescale data in the choice of our range. That is suppose you want to rescale your data points between 0 to 4(say) then you can easily do it by using MinMaxScaler. <br>
Let's implement it
`from sklearn import preprocessing 
import pandas as pd`

`data = pd.read_csv("restaurent.csv")
data`
![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/ob06a5uaf6txy7idussy.png)

![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/luq3coslshe7fw8wqoby.png)

![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/l9t5hp92wcki3n361qqj.png)

### Boom!! Normalization done using MinMaxScaler()
Note: The beauty of the approach is that we can normalize all the columns in one go. 

## Let's Normalise the data in range of our choice
This will be done with the help of MinMaxScaler only but this time we will change the feature_range to, say, (0,3) 

![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/pjtrezzeu7lwyu9c193i.png)

`data2 = pd.read_csv("restaurent.csv")
data`   
![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/suc9pbplk0o16svm7exr.png)

`normalizer=preprocessing.MinMaxScaler(feature_range=(0,3))
col_name=data.columns
d=normalizer.fit_transform(data)
normalized_data=pd.DataFrame(d,columns=col_name)
normalized_data` 

![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/hyk7y1ipf8i9xuwkojo2.png)

## Boom!! 
we can observe that it is rescaled to range of 0 to 3 for all the columns in just one go. 

## Conclusion 
As we have seen in the above tutorial that we can normalize data in the range of our choice with the help of MinMaxScaler() provided under sklearn. But it is always recommended to normalize data between zero to one. 

So, That's all from this tutorial. See you next tutorial blog. Thank you!

 
  
