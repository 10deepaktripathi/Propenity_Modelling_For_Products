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
