# Literature review

This directory contains the literature review and associated files.

## Contents

* [Survey papers](#survey-papers)
* [Topic papers](#topic-papers)
* [Books](#books)
* [Key terms](#key-terms)
* [Questions](#questions)
* [Thoughts](#thoughts)

## Survey papers

1. [Zhuang, 2020. A Comprehensive Survey on Transfer Learning](https://arxiv.org/pdf/1911.02685.pdf?ref=https://githubhelp.com). 

    Defines transfer learning and associated key terms.

1. [Tan, 2018. A Survey on Deep Transfer Learning](https://arxiv.org/pdf/1808.01974.pdf%3E).

    Defines and categorizes deep transfer learning.

1. [O'Neill, 2020. An overview of neural network compression](https://arxiv.org/pdf/2006.03669.pdf).

    Discusses common approaches for neural network compression, including
    weight sharing, pruning, decomposition, knowledge distillation, and
    quantization.

## Topic papers

1. [Yosinski, 2014. How transferable are features in deep neural networks?](https://proceedings.neurips.cc/paper/2014/file/375c71349b295fbe2dcdca9206f20a06-Paper.pdf)

    Quantifies layer generality/specificity--i.e., how well features in a given
    layer transfer between tasks.

1. [Knyazev, 2021. Parameter Prediction for Unseen Deep Architectures](https://arxiv.org/abs/2110.13100).

    Trains a graph hypernetwork to predict the parameters of a wide range of
    architectures on ImageNet and CIFAR-10.

1. [Brock, 2017. FreezeOut: Accelerate training by progressively freezing layers](https://arxiv.org/pdf/1706.04983.pdf?source=post_page---------------------------).

    Decreases the learning rate for and, ultimately, freezes layers according to
    a schedule, speeding up training marginally (20%) while preserving accuracy.

1. [Finn, 2017. Model-agnostic meta-learning for fast adaptation of deep networks](http://proceedings.mlr.press/v70/finn17a/finn17a.pdf).

    Introduces a meta-learning regime that trains a model to generalize to new
    tasks in a few-shot setting.

1. [Fey, 2021. GNNAutoScale: Scalable and Expressive Graph Neural Networksvia Historical Embeddings](https://cs.stanford.edu/~jure/pubs/gnnautoscale-icml21.pdf).

    Uses historical embeddings to enable the training of GNNs on very large
    graphs.

1. [Lucas, 2021. Analyzing Monotonic Linear Interpolation in Neural Network Loss Landscapes](http://proceedings.mlr.press/v139/lucas21a/lucas21a.pdf).

    Challenges the Monotonic Linear Interpolation property observed in previous
    research.

1. [Neyshabur, 2020. What is being transferred in transfer learning?](https://proceedings.neurips.cc/paper/2020/file/0607f4c705595b911a4f3e7a127b44e0-Paper.pdf).

    Classifies the knowledge transferred through model weights as "features" and
    "low-level statistics."

1. [Kornblith, 2019. Similarity of Neural Network Representations Revisited](http://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf).

    Introduces a similarity metric for comparing trained neural networks.

1. [He, 2019. Rethinking imageNet pre-training](https://openaccess.thecvf.com/content_ICCV_2019/papers/He_Rethinking_ImageNet_Pre-Training_ICCV_2019_paper.pdf).

    Challenges the paradigm in the computer vision community that one must use
    pretrained networks to achieve SOTA results.

1. [Wei, 2018. Transfer Learning via Learning to Transfer](http://proceedings.mlr.press/v80/wei18a/wei18a.pdf)

    Learns to perform feature-based transfer from previous transfer learning
    experiences.

1. [Wang, 2019. Characterizing and Avoiding Negative Transfer](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Characterizing_and_Avoiding_Negative_Transfer_CVPR_2019_paper.pdf)

    Provides a mathematical definition for negative transfer and proposes an
    improvement to adversarial feature extractors for transfer learning to avoid
    it.

1. [Devlin, 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf&usg=ALkJrhhzxlCL6yTht2BRmH9atgvKFxHsxQ)

    Introduces BERT, a pre-trained bidirectional transformer model that can be
    fine-tuned to achieve SOTA results on many NLP benchmarks.

1. [Chronopoulou, 2019. An embarrassingly simple approach for transfer learning from pretrained language models](https://arxiv.org/pdf/1902.10547.pdf)

    Adds a generic loss term during fine-tuning of language models to prevent
    models from forgetting task-agnostic features.

1. [Wong, 2018. Transfer Learning with Neural AutoML](https://proceedings.neurips.cc/paper/2018/file/bdb3c278f45e6734c35733d24299d3f4-Paper.pdf)

    Uses model-based deep transfer learning and task embeddings to design a
    neural architecture search model that rapidly converges on new tasks.

1. [Wang, 2020. K-ADAPTER: Infusing Knowledge into Pre-Trained Models with Adapters](https://arxiv.org/pdf/2002.01808.pdf)

    Introduces the adapter module, a plug-in subnetwork for transformer
    architectures, to improve performance in the multi-task NLP setting.

1. [Vaswani, 2017. Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

    Introduces the Transformer architecture based on stacked
    attention/self-attention layers.

1. [Frankle, 2019. The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/pdf/1803.03635.pdf?ref=https://githubhelp.com)

    Proposes the Lottery Ticket Hypothesis, according to which any sufficiently
    large, randomly initialized dense network contains a subnetwork that, when
    trained, matches the accuracy of the original network.

1. [Tolstikhin, 2021. MLP-Mixer: An all-MLP architecture for vision](https://proceedings.neurips.cc/paper/2021/file/cba0a4ee5ccd02fda0fe3f9a3e7b89fe-Paper.pdf)

    Demonstrates competitive (near-SOTA) results on ImageNet using only dense
    layers, with some additional transformations to achieve position-invariance.

1. [Wu, 2021. sciCvT: Introducing Convolutions to Vision Transformers](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_CvT_Introducing_Convolutions_to_Vision_Transformers_ICCV_2021_paper.pdf)

    Achieves SOTA performance on ImageNet by extending the Transformer
    architecture using convolutional token embeddings and convolutional
    projections for attention.

1. [Huang, 2022. Large Language Models Can Self-Improve](https://arxiv.org/pdf/2210.11610.pdf)

    Uses a pretrained large language model to generate Chain of Thought answers
    for unlabeled questions, then fine-tunes the model on high-confidence
    answers to achieve SOTA performance on several NLP benchmarks.

1. [Howard, 2018. Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/pdf/1801.06146.pdf)

    Proposes universal language model fine-tuning (ULMFiT), a procedure for
    reliable transfer learning on any NLP task.

1. [Shazeer, 2017. Outrageously Large Neural Networks: The Sparsely-gated Mixture-of-experts Layer](https://arxiv.org/pdf/1701.06538.pdf%22%20%5Ct%20%22_blank)

    Introduces a conditional layer that allows users to dramatically increase
    the capacity of distributed neural networks without an increase in
    computation.

## Books

1. [Yang, 2020. Transfer Learning](https://doi.org/10.1017/9781139061773)

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

* Categories of deep transfer learning [Tan, 2018. A Survey on Deep Transfer Learning](https://arxiv.org/pdf/1808.01974.pdf%3E):

    * Instance-based deep transfer learning: Based on instance weighting
    strategies; analogous to classical instance-based transfer learning.
    
    * Mapping-based deep transfer learning: Maps features into a new feature
    space with better similarity between source and target domain; analogous to
    classical feature-based transfer learning.
    
    * Network-based deep transfer learning: Reuses some or all of the pretrained
    source domain network; analogous to classical parameter-based transfer
    learning.
    
    * Adversarial-based: Uses adversarial technology to find transferable
    features between domains.

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

* Transfer learning training strategies [Yosinski, 2014. How transferable are features in deep neural networks?](https://proceedings.neurips.cc/paper/2014/file/375c71349b295fbe2dcdca9206f20a06-Paper.pdf):

    * Fine-tuning: When training on the new task, weight updates are allowed to
    backpropagate through the entire network, including layers whose weights
    were transferred from a pretrained model. If the target dataset is small,
    this process may result in overfitting; if the target dataset is
    sufficiently large, there is an expected performance gain over weight
    freezing.
    
    * Weight freezing: When training on the new task, the layers whose weights
    were transferred from a pretrained model are not updated. Using frozen
    features preserved generality.
    
    Note: Not all researchers make this distinction.

* Fragile co-adaptation: The phenomenon that features in successive layers of a
neural network interact with each other in a complex or fragile way such that
this co-adaptation cannot be relearned by the upper layers alone when lower
layers are frozen and transferred. Fragile co-adaptation tends to occur in the
middle of a network: in the lower layers, features are general and little
co-adaptation occurs; in the upper layers, there is less to learn if the network
is frozen from that point, so gradient descent can find a good solution. But in
the middle layers, freezing and transferring can cause significant performance
degradation. Fine-tuning appears to counteract fragile co-adaptation. [Yosinski, 2014. How transferable are features in deep neural networks?](https://proceedings.neurips.cc/paper/2014/file/375c71349b295fbe2dcdca9206f20a06-Paper.pdf)

## Questions

* The definitions of transfer learning presented in [Yang, 2020. Transfer Learning](https://doi.org/10.1017/9781139061773)
and [Tan, 2018. A Survey on Deep Transfer Learning](https://arxiv.org/pdf/1808.01974.pdf%3E)
state that transfer learning occurs when the source domain is not equal to the
target domain, or the source task is not equal to the target task. But what
about when both the domain and task are identical for source and target, but we
want to try a new model architecture on a fraction of the data or without
training from scratch? These definitions do not account for such a goal.

* Could embeddings, e.g., task embeddings, be considered parameter-based
transfer learning techniques?

## Thoughts

* The purpose of matrix multiplication `AB` is to transform the row vectors in
`A` to row vectors in a new subspace. If `A` is `m x n` and `B` is `n x o`, then
the columns of `B` describe the linear transformations from `n`-dimensional
space to `o`-dimensional space.
