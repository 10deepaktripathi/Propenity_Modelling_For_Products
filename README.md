# Propenity_Modelling_For_Wine_Company
In this project, we have created a recommendation model which upon giving a product
can generate a list of potential customers who might be interested in the product. Recom-
mendations are generated along with a confidence score which shows how relevant that
recommendation is. The data we have dealt with is implicit in nature as it just has the cus-
tomer’s purchasing history and we have explored various ways to deal with it. We explored
the Bayesian personalized ranking and ALS implicit recommendation model, which are spe-
cially designed to deal with implicit data and make good recommendations. Eventually, we
tried embedding-based two-tower deep learning models to build a recommendation system.
We have tried exploring different architectures of it such as MLP based, Matrix Factoriza-
tion based and the one, in which we have combined MLP and Matrix Factorization based
architectures together. We tried changing the objective function to check which objective
function (binary cross-entropy of right interaction or mean square error on frequency) gives
us better performance. To improve the performance, we tried engineering new features such
as recency, frequency, and monetary for both customers and products. Eventually, I compared
the result of these deep learning models with the popularity-based model, alternating least
squares and Bayesian personalized ranking model using the metric called HitRatio.

# Business Problem and Its Soluton
XYZ is one of the largest specialist retailer of wine. Occasionally
it has an excess inventory of wines that it wants to get rid of. The problem is how to
identify customers who might be interested in buying a particular wine. Once it has a list of
customers who might be interested in certain wines, it may contact/email those customers
and sell out that excess inventory. To attract more customers, it might also give targeted
offerings/discounts.
One of the solutions to this task is to create a propensity model for given categories of
products. This model will use customer’s historical behavior to predict their likelihood of
purchasing these products. That means the model will create, for every customer, a propensity
score for each of the agreed categories of product.
An appropriate way to accomplish this task is to create a recommendation model, which can
generate a list of customers, who might be interested in buying a given product. Along with
these recommendations, this model will also be generating a propensity score to show how
confident the system is in recommending each customer.

# Matrix factorization
Matrix factorization is one of the most popular approaches to implement collaborative
filtering. In matrix factorization, a large user-item sparse matrix is
decomposed into two smaller dense matrices using some MF techniques such as ALS,
SVD, or SVD++. One of these two matrices denotes only users in reduced vector space
in which each row relates to a specific user and another smaller matrix denotes only items
in the same vector space in which each row relates to a specific item. Once these smaller
matrices are learned, missing values can be predicted by multiplying these matrices again.

![Chart](charts/matrix_factorization.png)

The main goal while trying to learn smaller matrices is to minimize the below loss
function

![Chart](charts/formula_matrix_fact.png)


Where ru i is the actual rating, pu is a vector representing a specific user in smaller
vector space and qi is a vector representing a specific item in smaller vector space, λ is the
regularization hyper parameter, (∥p2u∥ + ∥q2i∥) is the regularization term with L2 norm.
This objective function can further be modified in various ways as stated in depending
on the data we are dealing with. In the paper, the author has suggested two approaches
namely SGD and ALS to minimize the above loss function.

# Implicit Data
As the name suggests, Implicit data is collected implicitly without asking the user to rate
the product explicitly. it is the purchases that a user makes, videos that he watches, or items
that he views or clicks. No matter which type of interaction is it, if a user has interacted with
an item, we call it positive feedback from the user given to that item even if the user has
interacted just once. Below is an image of implicit feedback interaction matrix Y

![Chart](charts/Implicit_data.png)

Where yu i shows the interaction between user and item. yu i is equal to 1 shows the user liked
the item, whereas yu i equal to 0 does not show any preference about the user. Note with this
formulation, if a user has not interacted with an item, it does not mean s/he disliked the item.
It might be that the user was not aware of this item. Similarly, if a user has interacted with an
item just once we are not sure if s/he has actually liked the item. Because of this uncertainty
in user preference, the matrix factorization technique which we discussed above will give
poor result with implicit data.


# Evaluation Techniques
You can use various techniques to come up with multiple recommendation models, but the
selection of one among them requires some sort of evaluation. There are various techniques
that nowadays can be used to evaluate the recommendation model performance offline,
among them I have described one below-

## HitRate@rankN:
the process of calculating the hit rate is as below[11]-
For each user, first, we remove an item with which the user has actually interacted and use
the remaining interactions for the training recommendation system. Once trained, use the
recommendation model to recommend N items for each user out of the total items available
in the database including the one which you had earlier taken out. If taken away item is
inside top n recommendations for a user, then it is a hit for that user. we then sum all the hits
and divide that by the number of users. The obtained number is called the hit rate.

# Dataset
The data I am going to work on has 3 important data files. One file contains all the information
related to customers, another file contains all the information related to products and the third
file contains all the detail related to any transactions that have been made by any customer
on any product. There are a total of 8 million transactions for 250,000 customers and 6600
products. Transnational data has the same customer and product combination multiple
times but on different timestamps showing that a customer has bought the same product
multiple times. I have pre-processed the transnational data to have a unique customer and
product combination with the information of a total number of times this product was bought
by the customer. Note that the same customer and same product might occur multiple
times individually in the pre-processed dataset, but the combination of them will be unique
across the dataset. I have created few two-tower model architectures in which I have used
customer and product context information as well and to do that I have merged pre-processed
transnational dataset obtained above with the customer file and product file on the basis of
customer code and product code respectively.


# Models
Note that we need to create a recommendation model, which does not only recommends
costumers given an input product but also generates a propensity(confidence) score between
0 to 1. I first have started implementing ALS Implicit, BPR, and Logistic Matrix Factoriza-
tion(LMF) models. Python already has an implementation of all of these models. The only
problem in using these models is that they do not generate confidence score between 0 to 1.
Still i will use all of these models for performance comparison. I then have started building
two-tower deep learning recommendation models. The advantage of using these models is
that if we use sigmoid activation function at the last layer of them, they will generate a score
between 0 to 1, which we can consider as confidence score. To verify how good these two
tower models are, i have verified their performance against ALS, BPR, LMF and popularity
based models. Below i have discussed in detail about all the models that i have tried.
