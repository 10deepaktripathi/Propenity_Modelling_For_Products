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

IMAGE

The main goal while trying to learn smaller matrices is to minimize the below loss
function

IMAGE


Where ru i is the actual rating, pu is a vector representing a specific user in smaller
vector space and qi is a vector representing a specific item in smaller vector space, λ is the
regularization hyper parameter, (∥p2u∥ + ∥q2i∥) is the regularization term with L2 norm.
This objective function can further be modified in various ways as stated in depending
on the data we are dealing with. In the paper, the author has suggested two approaches
namely SGD and ALS to minimize the above loss function.
