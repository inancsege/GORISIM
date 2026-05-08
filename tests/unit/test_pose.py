import torch
from gorisim.sign_to_text.pose import merge_hm


def test_merge_hm_takes_mean_across_scales():
    # 2 scales, 2 flip dims, 133 joints, 4x4 spatial
    a = torch.ones(2, 133, 4, 4)
    b = torch.ones(2, 133, 4, 4) * 3.0
    out = merge_hm([a, b])
    assert out.shape == (133, 4, 4)
    # mean of 1 and 3 = 2 (after flip merge stays 1 for a, 3 for b, then mean)
    assert torch.allclose(out, torch.full((133, 4, 4), 2.0), atol=1e-5)


def test_merge_hm_uses_concatenated_tensor_not_loop_var():
    """Original code had a bug where it averaged the last loop var only.
    With unequal-magnitude inputs the buggy version would equal the last input."""
    a = torch.zeros(2, 133, 4, 4)
    b = torch.full((2, 133, 4, 4), 10.0)
    out = merge_hm([a, b])
    # If buggy: out == b.mean(0) == 10. Correct: ((0+10)/2 averaged across all = 5
    assert torch.allclose(out.mean(), torch.tensor(5.0), atol=1e-5)
