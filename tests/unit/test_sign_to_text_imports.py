def test_imports_succeed():
    from gorisim.sign_to_text import utils  # noqa: F401
    from gorisim.sign_to_text.models import (
        pose_hrnet,  # noqa: F401
        resnet2plus1d,  # noqa: F401
    )
    from gorisim.sign_to_text.models.hrnet_config import cfg  # noqa: F401
