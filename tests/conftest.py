import pytest
from strands.models.model import Model


class FakeModel(Model):
    """Minimal Strands Model double: constructs an Agent without any API call.

    Never actually streamed in tests — the tests that use it only assert tool
    registration via Agent.tool_names, which does not invoke the model.
    """

    @property
    def stateful(self) -> bool:
        return False

    def update_config(self, **kwargs) -> None:  # pragma: no cover - inert
        pass

    def get_config(self) -> dict:  # pragma: no cover - inert
        return {}

    def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):  # pragma: no cover - inert
        raise NotImplementedError

    def stream(self, *args, **kwargs):  # pragma: no cover - inert
        raise NotImplementedError


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()
