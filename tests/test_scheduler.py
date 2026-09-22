import pytest
import torch

def test_scheduled_changes_lr_at_milestones():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,6], gamma =0.5)
    lrs = []
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])
    assert lrs[1] == pytest.approx(1e-3)
    assert lrs[3] == pytest.approx(5e-4)
    assert lrs[6] == pytest.approx(2.5e-4)