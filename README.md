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

## Usage

**Note: This project is still in development, so not all of the functionality shown below may be implemented yet.**

### Setting the weights for an arbitrary model on an arbitrary task

We would like to set the weights of a new model of arbitrary architecture to maximize its accuracy on an arbitrary
dataset. We use `dogwood.get_pretrained_model(model, X_train, y_train)` to find the best weights for the given
architecture and learning task based on a store of trained models, including popular ones like VGG, BERT, and StyleGAN.

```python
import numpy as np
from tensorflow.keras.models import Model
import dogwood


def get_my_dataset() -> tuple[tuple[np.ndarray, np.ndarray],
                              tuple[np.ndarray, np.ndarray]]:
    # Your code here to return arbitrary (X_train, y_train), (X_test, y_test).
    pass


def get_my_model() -> Model:
    # Your code here to return a model with arbitrary architecture.
    pass


(X_train, y_train), (X_test, y_test) = get_my_dataset()
model = get_my_model()
print(f'Accuracy on arbitrary task/model before pretraining: '
      f'{model.evaluate(X_test, y_test)}') # Accuracy: 0.5
model = dogwood.get_pretrained_model(model, X_train, y_train)
print(f'Accuracy on arbitrary task/model after pretraining: '
      f'{model.evaluate(X_test, y_test)}') # Accuracy: 0.9
```

Output:

```
Accuracy on arbitrary task/model before pretraining: 0.5
Accuracy on arbitrary task/model after pretraining: 0.9
```

### Adding a trained model to the pretraining pool

By default, `dogwood` transfers weights from popular open source models, but we can also add models to the pool to make
learning on similar models/tasks even faster. Notice that this time we call
`pool.get_pretrained_model(model, X_train, y_train)` instead of `dogwood.get_pretrained_model(model, X_train, y_train)`.
The behavior of both is identical, but explicitly declaring the `PretrainingPool` object allows us to set its directory
to wherever we would like to keep our trained models.

```python
pool = dogwood.PretrainingPool(dirname='/path/to/my/pretraining/dir')
(X_train, y_train), (X_test, y_test) = get_my_dataset()
model = get_my_model()
model = pool.get_pretrained_model(model, X_train, y_train)
print(f'Accuracy when pretrained on default models: '
      f'{model.evaluate(X_test, y_test)}') # Accuracy: 0.9
model.fit(X_train, y_train, epochs=10)
print(f'Accuracy after fine-tuning: '
      f'{model.evaluate(X_test, y_test)}') # Accuracy: 0.95
pool.add_model(model, X_train, y_train)
model = get_my_model()
model = pool.get_pretrained_model(model, X_train, y_train)
print(f'Accuracy when pretrained on new models: '
      f'{model.evaluate(X_test, y_test)}') # Accuracy: 0.95
```

Output:

```
Accuracy when pretrained on default models: 0.9
Accuracy after fine-tuning: 0.95
Accuracy when pretrained on new models: 0.95
```

### Intended workflow for model prototyping

With the above functionality to load the best weights from pretrained models and add our own models to the pool, we can
design a model prototyping workflow that significantly reduces the cost in time and compute of training new model
architectures.

```python
# Create the model pool and dataset.
pool = dogwood.PretrainingPool(dirname='/path/to/my/pretraining/dir')
(X_train, y_train), (X_test, y_test) = get_my_dataset()

# Prototype the first model.
# Weights are set based on default open source pretrained models.
prototype_model_1 = Model(
    # Arbitrary architecture here.
)
prototype_model_1 = pool.get_pretrained_model(
    prototype_model_1, X_train, y_train)
prototype_model_1.fit(X_train, y_train, epochs=10)
pool.add_model(prototype_model_1, X_train, y_train)

# Prototype the second model.
# Weights are set from default models and all previously trained models.
# Training is much faster, because we are building on past knowledge.
prototype_model_2 = Model(
    # Arbitrary architecture here.
)
prototype_model_2 = pool.get_pretrained_model(
    prototype_model_2, X_train, y_train)
prototype_model_2.fit(X_train, y_train, epochs=10)
pool.add_model(prototype_model_2, X_train, y_train)

# Prototype the third model.
# ...
```

### Limitations

`dogwood.get_pretrained_model(model, X_train, y_train)` can only make `model` as performant as its architecture allows.
If `model` has an architecture that is inherently unsuited to its task, `dogwood` cannot make it achieve exceptional
results.
