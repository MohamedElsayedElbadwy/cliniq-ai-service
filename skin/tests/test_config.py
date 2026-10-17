from skin.config import get_settings, ARTIFACTS_DIR


def test_empty_optional_paths_use_defaults(monkeypatch):
    monkeypatch.setenv('SKIN_MODEL_PATH','')
    monkeypatch.setenv('SKIN_MANIFEST_PATH','')
    settings=get_settings()
    assert settings.model_path==ARTIFACTS_DIR/'skin_six.keras'
    assert settings.manifest_path==ARTIFACTS_DIR/'model_manifest.json'


def test_default_artifacts_do_not_load_previous_skin_experiment(monkeypatch):
    monkeypatch.delenv('SKIN_MODEL_PATH', raising=False)
    monkeypatch.delenv('SKIN_MANIFEST_PATH', raising=False)
    settings = get_settings()
    assert settings.model_path.parent.name == 'six-class'
    assert settings.manifest_path.parent == settings.model_path.parent
