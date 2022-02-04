# Set2Set 

An implementation of the paper [Order Matters: Sequence to Sequence for Sets](https://arxiv.org/abs/1511.06391v4).

## Usage

### Installation

```
git clone https://github.com/arunppsg/Set2Set.git & cd set2set
python setup.py install
```

### Example
```
import torch
import set2set as s2s
X = torch.randn(2, 5, 10)
model = s2s.Set2Set(10, 3, 1)
embedding = model(X)
```
