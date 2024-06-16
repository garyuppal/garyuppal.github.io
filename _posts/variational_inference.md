
*In this post, I introduce the basic ideas and motivation behind variational inference methods and give an overview of the different types of variational inference. In future posts, I go through in detail each class of variational inference method and work out example(s)...*

Introduction
--------------

Variational inference methods, also sometimes called variational Bayes methods, are a set of methods for approximating a posterior probability distribution that is otherwise difficult to compute exactly. 

The goal of Bayesian inference ... (see other post...) ...opposed to eg MCMC, HMC, EM, ..., pose as an optimization problem

The posterior distribution,
$$ p(z|x) = \frac{p(x|z)p(z)}{p(x)} $$
...given data x and varaibles z... approx with separate distributions. ...See separate post on Bayes theorem...

Problem statement
--------------
The posterior distribution \\(p(z|x)\\)is approximated by a variational distribution \\(q(z)\\). The goal is to then "push" \\(q(z)\\) to be as close as possible to \\(p(z|x)\\). This can be expressed quantitatively as wanting to minimize the KL divergence between \\(p(z|x)\\) and \\(q(z)\\). The KL divergence between two distributions \\(Q(x)\\) and \\(P(x)\\) is given as
$$ D_{KL}(Q||P) \triangleq \int_x Q(x) \log \frac{Q(x)}{P(x)} $$

*Note, this isn't the only measure of divergence, and we can alternatively express the distance between the distributions in another way*... note also that this is not symmetric ...why is this what is chosen then? ...see wikipedia, connections between EM and expectation propagation??....


Variational inference methods
--------------

The basic idea is as given up. details/choices are in how the approximate posterior is selected and how parameters are optimizted... cavi vs black box vs amortized, etc...

...links to different methods, each of which should have a tutorial(s) on some basic examples...

Pitfalls, challenges, tips...
--------------

what to be aware of when using methods..., some intuition and advice?...

- using KL divergence, often tend to underestimate the true posterior variance
- mean field can be poor approximation and can miss interdependence/correlations

Resources ...References?
--------------
...papers, youtube and article links... (way to cite in above?)
