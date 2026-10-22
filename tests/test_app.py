from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def test_app_renders_without_errors(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHROMA_DB_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("OPENAI_API_KEY", "")

    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value.startswith("📚 AI Study Assistant")
    assert [tab.label for tab in app.tabs] == [
        "1. Process notes",
        "2. Inspect data",
        "3. Ask questions",
    ]
