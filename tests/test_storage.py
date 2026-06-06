from wildlife_reid import storage


def test_is_remote():
    assert storage.is_remote("gs://bucket/path")
    assert not storage.is_remote("artifacts/index")
    assert not storage.is_remote("/abs/local/path")


def test_join_uri_remote():
    assert storage.join_uri("gs://bucket/sea_turtle", "index") == "gs://bucket/sea_turtle/index"
    # trailing/leading slashes are normalized
    assert storage.join_uri("gs://bucket/sea_turtle/", "/index/") == "gs://bucket/sea_turtle/index"


def test_join_uri_local():
    joined = storage.join_uri("artifacts/sea_turtle", "index")
    assert joined.replace("\\", "/") == "artifacts/sea_turtle/index"


def test_write_json_local(tmp_path):
    target = tmp_path / "nested" / "manifest.json"
    storage.write_json(target, {"gallery_size": 3})
    import json

    assert json.loads(target.read_text())["gallery_size"] == 3


def test_read_json_local(tmp_path):
    target = tmp_path / "manifest.json"
    target.write_text('{"gallery_size": 5}', encoding="utf-8")
    assert storage.read_json(target)["gallery_size"] == 5
