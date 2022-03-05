# Literature review

This directory contains the literature review and associated files.

## Contents

* [Survey papers](#survey-papers)
* [Topic papers](#topic-papers)
* [Key terms](#key-terms)
* [Questions](#questions)
* [Thoughts](#thoughts)

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
    most interested in parameter-based transfer learning. Many of the methods
    discussed add regularizers for training the target model from scratch,
    rather than devising methods to shorten training time by starting with some
    pretrained weights. We may be able to use parameter restriction techniques
    (as opposed to parameter sharing) in the deep transfer setting. Due to
    differences in source/target distributions, directly combining data or
    models may not be successful, which is why some researchers use data-based
    approaches to transfer learning or model ensembling. Deep learning
    approaches to transfer learning include stacked autoencoders (SDA, mSLDA,
    TLDA) and adversarial deep learning (GAN--although not sure this really
    counts, DANN). Deep learning approaches often add a distribution adaptation
    loss term/regularizer to minimize the "distance" between the intermediate
    (i.e., latent) representations of the source and target distributions.
    
    Contains good general insights, but focused on transfer learning in the
    traditional ML setting, not really deep transfer learning. Notes that
    adversarial learning is very powerful in deep transfer learning based on
    experimental results.
    
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

2. [Luo, 2008. Transfer learning from multiple source domains via consensus regularization](https://dl.acm.org/doi/pdf/10.1145/1458082.1458099?casa_token=AgWZfrnGhVsAAAAA:BWj-fjIN38cM5eQ6OXJQJqOKub0KnxsQFCf2hOjnBsOG4fiYE2N5OBovYLwusTVSeEviFxiIPtk).

    TODO

3. Tzeng, 2014. Deep domain confusion: Maximizing for domain invariance.

    TODO Explores "adaptation layers" and a "discrepancy loss" for training
    autoencoders for domain transfer.

4. Masqood, 2019. Transfer learning assisted classification and detection of
Alzheimer's disease stages using 3D MRI scans.

    TODO Recent example of deep transfer learning techniques in a biomedical
    setting. The biomedical aspect is not what is interesting; this is a recent
    example of how deep transfer learning is actually used in SOTA applications.

5. Wang, 2019. Characterizing and avoiding negative transfer.

    TODO

6. Lipton, 2018. The mythos of model interpretability.

    TODO

7. [Hu, 2018. Exploring Weight Symmetry in Deep Neural Networks](https://arxiv.org/abs/1812.11027).

    TODO Finds that, in some convolutional and recurrent architectures, weight
    symmetry does not significantly reduce model performance as might be
    supposed. At the same time, weight symmetry reduces model size by 25%. Do
    these results extend to all architectures? Do we need to worry about weight
    symmetry at all?
    

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

* Marginal distribution: The probability `P(X)` of any given combination of
feature values occurring. Inferred from [Zhuang, 2020. A Comprehensive Survey on Transfer Learning](https://arxiv.org/pdf/1911.02685.pdf?ref=https://githubhelp.com).

* Conditional distribution: The probability `P(Y | X)` of labels `Y` conditioned
on the instance set of features `X`. Notably, different marginal distributions
`P(X1)` and `P(X2)` could still have the same conditional distribution, e.g.,
because of differences in sampling (suppose `X1` is the set of features drawn
from young people and `X2` is the set of features drawn from elderly people; the
conditional distribution `P(Y | X)` measuring the likelihood that a person has
cancer is the same regardless of the differences between `P(X1)` and `P(X2)`).
Inferred from [Zhuang, 2020. A Comprehensive Survey on Transfer Learning](https://arxiv.org/pdf/1911.02685.pdf?ref=https://githubhelp.com).

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

    * Instance-based: Based on instance weighting strategies. This is most often
    used when the source and target domains differ only in marginal
    distributions, so the examples from the source domain most relevant to the
    target domain are weighted higher in the target learner's loss function.
    
    * Feature-based: Transform the original features to create a new feature
    representation. Asymmetric approaches transform source features to match the
    target ones; symmetric approaches attempt to find a common latent feature
    space. The goal is to minimize the marginal and conditional distribution
    difference between source and target domain instances.
    
    * Parameter-based: Transfer knowledge at the model/parameter level. This
    category includes parameter sharing, which is a key technique in deep
    transfer learning.
    
    * Relational-based: Focus on problems in relational domains; transfer the
    logical relationship or rules learned in the source domain to the target
    domain.

* Partial transfer learning: Approaches designed for the scenario in which the
target domain classes are a subset of the source domain classes. TODO this
description could be wrong based on the description in the paper. See #17. [Zhuang, 2020. A Comprehensive Survey on Transfer Learning](https://arxiv.org/pdf/1911.02685.pdf?ref=https://githubhelp.com).

* Reinforcement transfer learning: TODO

* Lifelong transfer learning: TODO

* Online transfer learning: TODO

* Kernel trick: TODO

* Incremental learning: TODO

* Consensus regularizer: An additional loss term that promotes agreement between
two or more models, often in the form of cross-entropy. [Luo, 2008. Transfer learning from multiple source domains via consensus regularization](https://dl.acm.org/doi/pdf/10.1145/1458082.1458099?casa_token=AgWZfrnGhVsAAAAA:BWj-fjIN38cM5eQ6OXJQJqOKub0KnxsQFCf2hOjnBsOG4fiYE2N5OBovYLwusTVSeEviFxiIPtk).

* Parameter sharing: A method of transferring the knowledge from a source model
to a target model in which the target model freezes (i.e., shares) some of the
source model parameters, and only finetunes a portion (in deep transfer
learning, traditionally the last few layers of the neural network). In
traditional ML, matrix factorization is a popular strategy. [Zhuang, 2020. A Comprehensive Survey on Transfer Learning](https://arxiv.org/pdf/1911.02685.pdf?ref=https://githubhelp.com)

* Parameter restriction: A method of transferring the knowledge from a source
model to a target model in which the target model uses similar, but not
necessarily identical weights as the source model; a variant on parameter
sharing. [Zhuang, 2020. A Comprehensive Survey on Transfer Learning](https://arxiv.org/pdf/1911.02685.pdf?ref=https://githubhelp.com)

## Questions

* If transfer learning is defined as "improving the performance of target
learners on target domains by transferring the knowledge contained in different
but related source **domains**," does that include improving the performance of
target learners on target domains by transferring the knowledge contained in
different but related source **learners**? In our work, we aim to do both, i.e.,
improve the performance of target learners on target domains by transferring the
knowledge contained in different but related source **learners and/or domains**.

## Thoughts

* The purpose of matrix multiplication `AB` is to transform the row vectors in
`A` to row vectors in a new subspace. If `A` is `m x n` and `B` is `n x o`, then
the columns of `B` describe the linear transformations from `n`-dimensional
space to `o`-dimensional space.
