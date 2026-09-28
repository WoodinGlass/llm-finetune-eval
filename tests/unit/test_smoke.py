"""Smoke tests — must pass without GPU, network, or external services."""
from pathlib import Path


def test_repo_layout() -> None:
    for p in ["training", "eval", "configs", "serving", "tests", "docs"]:
        assert Path(p).is_dir(), f"missing dir: {p}"


def test_slo_exists() -> None:
    assert Path("docs/slo.md").is_file()


def test_env_example_has_required_keys() -> None:
    text = Path(".env.example").read_text()
    for key in ["HF_TOKEN", "WANDB_API_KEY", "SERVING_API_KEY"]:
        assert key in text, f"missing {key} in .env.example"


def test_pyproject_has_extras() -> None:
    import tomllib
    data = tomllib.loads(Path("pyproject.toml").read_text())
    extras = data["project"]["optional-dependencies"]
    for name in ["dev", "data", "train", "eval", "serve", "observability"]:
        assert name in extras, f"missing extra: {name}"
