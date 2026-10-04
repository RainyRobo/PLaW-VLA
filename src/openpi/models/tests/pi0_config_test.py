import pytest

from openpi.models import pi0_config
from openpi.models import vjepa2 as _vjepa2


@pytest.mark.parametrize("variant", _vjepa2.known_variants())
def test_pi0_config_accepts_canonical_vjepa2_variants(variant):
    config = pi0_config.Pi0Config(vjepa2_variant=variant)

    assert config.vjepa2_variant == variant


def test_pi0_config_rejects_legacy_vjepa2_variant_names():
    with pytest.raises(ValueError, match="Unknown vjepa2_variant"):
        pi0_config.Pi0Config(vjepa2_variant="vith-fpc64-256")
