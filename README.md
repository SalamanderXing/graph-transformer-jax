# JAX Graph Transformer

This is a JAX/Flax version of [Lucidrains'](https://github.com/lucidrains/graph-transformer-pytorch) Graph Transfomer in PyTorch. Notice this implementation does not yet support positional embeddings.

```python
from jax import random

from graph_transformer import GraphTransformer


model, params = GraphTransformer.initialize(
    random.PRNGKey(0),
    number_of_nodes=9,
    num_layers=2,
    in_edge_features=10,
    in_node_features=10,
)

print(model)
```
