# Literature review

This directory contains the literature review and associated files.

## Contents

* [Survey papers](#survey-papers)
* [Topic papers](#topic-papers)
* [Key terms](#key-terms)
* [Questions](#questions)

## Survey papers

1. [Zhuang, 2020. A Comprehensive Survey on Transfer Learning](https://arxiv.org/pdf/1911.02685.pdf?ref=https://githubhelp.com). 

    Defines transfer learning in terms of reducing the data required to succeed
    in a new domain, although we also want to focus on transferring the
    knowledge to a different architecture (the domain may or may not change).
    When we say that our research aims to make transfer learning the default
    posture for any training regime, perhaps what we are really saying is that
    we want to reduce the time (and hence data) required to succeed in the new
    domain and/or new architecture to the minimum. So this definition is
    consistent with our objectives. Semi-supervised learning is an alternative,
    but requires that you can collect large amounts of unlabeled data in
    addition to some labeled data. It is possible to achieve better results by
    transferring knowledge from/to more than one domain. It seems that we are
    most interested in parameter-based transfer learning.
    
    Suggested future work: measuring the transferability across domains and
    avoiding negative transfer; interpretability of transfer learning
    techniques.

2. [Tan, 2018. A Survey on Deep Transfer Learning](https://arxiv.org/pdf/1808.01974.pdf%3E).

    TODO

3. [Pan, 2010. A Survey on Transfer Learning](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.147.9185&rep=rep1&type=pdf).

    TODO this is one of the foundational papers.

4. [Wang, 2018. Deep visual domain adaptation: A survey](https://arxiv.org/pdf/1802.03601.pdf?ref=https://githubhelp.com)

    TODO although this is focused on vision systems, the deep learning aspect
    could be very informative.

## Topic papers

1. [Knyazev, 2021. Parameter Prediction for Unseen Deep Architectures](https://arxiv.org/abs/2110.13100).

    TODO

## Key terms

* Transfer learning: A machine learning technique that aims at improving the
performance of target learners on target domains by transferring the knowledge
contained in different but related source domains. Given some/an observation(s)
corresponding to `m^S in N^+` (i.e., some positive number) source domain(s) and
task(s), and some/an observation(s) about `m^T in N^+` target domain(s) and
task(s), transfer learning utilizes the knowledge implied in the source
domain(s) to improve the performance of the learned decision function(s) `f^Tj`
(`j = 1, ..., m^T`) on the target domain(s). [Zhuang, 2020. A Comprehensive Survey on Transfer Learning](https://arxiv.org/pdf/1911.02685.pdf?ref=https://githubhelp.com).

* Feature space: A feature space `Chi` is the cartesian product of the range of
values for all features. Inferred from [Zhuang, 2020. A Comprehensive Survey on Transfer Learning](https://arxiv.org/pdf/1911.02685.pdf?ref=https://githubhelp.com).

* Domain: A domain `D` is composed of a feature space `Chi` and a marginal
distribution `P(X)`, i.e., `D = {Chi, P(X)}`. The symbol `X` denotes an instance
set, which is defined as `X = {x | xi in Chi, i = 1, ..., n}`. In other words,
the feature space is the cartesian product of the range of values for all
features, `X` is the set of observations in the feature space, and `P(X)` is the
probability of any given combination of feature values. [Zhuang, 2020. A Comprehensive Survey on Transfer Learning](https://arxiv.org/pdf/1911.02685.pdf?ref=https://githubhelp.com).

* Task: A task `Tau` consists of a label space `Upsilon` and a decision function
`f`, i.e., `Tau = {Upsilon, f}`. The decision function `f` is implicit and is
expected to be learned from the sample data. `f` is a function from `Chi` to
`Tau`. [Zhuang, 2020. A Comprehensive Survey on Transfer Learning](https://arxiv.org/pdf/1911.02685.pdf?ref=https://githubhelp.com).

* Negative transfer: The phenomenon that previous experience has a negative
effect on learning new tasks, e.g., knowing Spanish interferes with learning
French. [Zhuang, 2020. A Comprehensive Survey on Transfer Learning](https://arxiv.org/pdf/1911.02685.pdf?ref=https://githubhelp.com).

* Transfer learning domain settings:

    * Homogeneous transfer learning: A category of transfer learning in which
    the source and target domains are of the same feature space. Distribution
    adaptation methods are used to bridge the gap between domains. As opposed to
    heterogeneous transfer learning. [Zhuang, 2020. A Comprehensive Survey on Transfer Learning](https://arxiv.org/pdf/1911.02685.pdf?ref=https://githubhelp.com).

    * Heterogeneous transfer learning: A category of transfer learning in which
    the source and target domains are of different feature spaces. In addition
    to distribution adaptation, this category of learning requires feature space
    adaptation. As opposed to homogeneous transfer learning. [Zhuang, 2020. A Comprehensive Survey on Transfer Learning](https://arxiv.org/pdf/1911.02685.pdf?ref=https://githubhelp.com).

* Fields related to transfer learning [Zhuang, 2020. A Comprehensive Survey on Transfer Learning](https://arxiv.org/pdf/1911.02685.pdf?ref=https://githubhelp.com):

    * Semi-supervised learning: A machine learning task in which only some of
    the data are labeled. In contrast to transfer learning, both labeled and
    unlabeled data are drawn from the same distribution; in transfer learning,
    source and target domains are usually different. Transfer learning does,
    however, use some of the same assumptions as semi-supervised learning, e.g.,
    smoothness, cluster, and manifold assumptions.

    * Multi-view learning: A machine learning task in which a learner's
    performance is improved by considering objects from multiple views, e.g.,
    from a video source and an audio source.

    * Multi-task learning: A machine learning task in which a learner trains
    jointly on a group of related tasks.

* Domain adaptation: The process of adapting one or more source domains to
reduce the difference with the target domain for the purposes of improving
target learner performance. [Zhuang, 2020. A Comprehensive Survey on Transfer Learning](https://arxiv.org/pdf/1911.02685.pdf?ref=https://githubhelp.com).

* Transfer learning label settings [Pan, 2010. A Survey on Transfer Learning](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.147.9185&rep=rep1&type=pdf):

    * Transductive transfer learning: A label setting in which label information
    comes only from the source domain.

    * Inductive transfer learning: A label setting in which label information
    from the target domain is available.

    * Unsupervised transfer learning: A label setting in which label information
    is unknown for both source and target domain.

* Transfer learning solution categories [Pan, 2010. A Survey on Transfer Learning](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.147.9185&rep=rep1&type=pdf):

    * Instance-based: Based on instance weighting strategies.
    
    * Feature-based: Transform the original features to create a new feature
    representation. Asymmetric approaches transform source features to match the
    target ones; symmetric approaches attempt to find a common latent feature
    space.
    
    * Parameter-based: Transfer knowledge at the model/parameter level.
    
    * Relational-based: Focus on problems in relational domains; transfer the
    logical relationship or rules learned in the source domain to the target
    domain.

* Reinforcement transfer learning: TODO

* Lifelong transfer learning: TODO

* Online transfer learning: TODO

## Questions

* If transfer learning is defined as "improving the performance of target
learners on target domains by transferring the knowledge contained in different
but related source **domains**," does that include improving the performance of
target learners on target domains by transferring the knowledge contained in
different but related source **learners**? In our work, we aim to do both, i.e.,
improve the performance of target learners on target domains by transferring the
knowledge contained in different but related source **learners and/or domains**.
