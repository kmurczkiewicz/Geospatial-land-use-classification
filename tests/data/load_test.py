import pytest

import src.data.load


@pytest.mark.skip(reason="Loads dict of images as np arrays into memory (around 2.5 GB), not testable currently")
def load_into_memory():
    pass
