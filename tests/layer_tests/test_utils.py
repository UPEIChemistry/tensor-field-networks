from tfn.utils import cartesian_loss, manhattan_loss


def test_cartesian_mae(random_z_and_cartesians):
    _, c = random_z_and_cartesians
    loss = cartesian_loss(c, c + 1.0)
    assert loss.shape == c.shape[:1]


def test_manhattan_loss(random_z_and_cartesians):
    _, c = random_z_and_cartesians
    loss = manhattan_loss(c, c + 1.0)
    assert loss.shape == c.shape[:1]
