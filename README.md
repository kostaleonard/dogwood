# dogwood

Leo's PhD repository.

## Installation

```bash
pip install dogwood
```

## Motivation

Building on past knowledge should be the default behavior of every neural network, regardless of architecture or
learning task. Engineers and researchers waste significant time and computational resources trying to reproduce the
results of already-published models, even when working on identical architectures and tasks. When a developer creates a
new model, it should automatically set its parameters to maximize performance based on known models and tasks. If
architecture and task are nearly identical, then the performance of the model should be at least as good as the previous
best model; if the architecture and/or task differ significantly, then the model should distill knowledge from past runs
to achieve superior performance.

Training a model from scratch is still a valid strategy for some applications, but such a regime should be the result of
a developer's explicit decision to deviate from transfer-learning-by-default.

**Vision: Unless a developer specifically decides to train from scratch, every new model should be at least as good as
the previous best performing model of similar, but not necessarily identical, architecture.**

## Literature review

For a complete list of references used, please see the [project literature review](literature/README.md).
