import pytest
import set2set as s2s

def test_set2set_model():
    model = s2s.Set2Set(10, 3, 1)
    assert repr(model) == 'Set2Set(10, 20)'

    x = torch.randn(2, 5, 10)
    out = model(x)
    assert out.size() == (2, 20)
