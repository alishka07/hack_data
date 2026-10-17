"""Pinned verifier downloads never publish incomplete or unverified weights."""
import hashlib
import io
import json

import pytest

import scripts.setup_verifier_gguf as setup


class Response(io.BytesIO):
    def __init__(self, content, status=200, headers=None):
        super().__init__(content)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(content))}


def asset(content=b"verified synthetic model"):
    return setup.Asset("synthetic.gguf", len(content), hashlib.sha256(content).hexdigest(), "https://example.invalid/pinned", "model")


def test_download_is_verified_before_final_name_and_reused_without_network(tmp_path):
    content = b"verified synthetic model"
    expected = asset(content)
    final = tmp_path / expected.filename
    def opener(request, timeout):
        assert not final.exists()
        return Response(content)
    assert setup._ensure_asset(expected, final, opener=opener) is True
    assert final.read_bytes() == content
    assert not final.with_name(final.name + ".part").exists()
    assert setup._ensure_asset(expected, final, opener=lambda *a, **k: pytest.fail("Unexpected network")) is False


def test_partial_resume_checks_exact_range_and_publishes_only_complete_hash(tmp_path):
    content = b"verified synthetic model"
    expected = asset(content)
    final = tmp_path / expected.filename
    final.with_name(final.name + ".part").write_bytes(content[:7])
    def opener(request, timeout):
        assert request.get_header("Range") == "bytes=7-"
        return Response(content[7:], 206, {"Content-Length": str(len(content) - 7),
                                         "Content-Range": f"bytes 7-{len(content)-1}/{len(content)}"})
    assert setup._ensure_asset(expected, final, opener=opener)
    assert final.read_bytes() == content


def test_server_ignoring_range_restarts_partial_instead_of_appending(tmp_path):
    content = b"verified synthetic model"
    expected = asset(content)
    final = tmp_path / expected.filename
    final.with_name(final.name + ".part").write_bytes(content[:7])
    assert setup._ensure_asset(expected, final, opener=lambda *a, **k: Response(content))
    assert final.read_bytes() == content


def test_wrong_hash_leaves_existing_final_untouched_and_discards_bad_partial(tmp_path):
    content = b"verified synthetic model"
    expected = asset(content)
    final = tmp_path / expected.filename
    final.write_bytes(b"older final")
    with pytest.raises(ValueError, match="SHA-256"):
        setup._ensure_asset(expected, final, opener=lambda *a, **k: Response(b"x" * len(content)))
    assert final.read_bytes() == b"older final"
    assert not final.with_name(final.name + ".part").exists()


def test_short_transfer_stays_partial_and_can_resume_later(tmp_path):
    content = b"verified synthetic model"
    expected = asset(content)
    final = tmp_path / expected.filename
    with pytest.raises(OSError, match="ended early"):
        setup._ensure_asset(expected, final, opener=lambda *a, **k: Response(content[:5], headers={"Content-Length": str(len(content))}))
    assert not final.exists()
    assert final.with_name(final.name + ".part").read_bytes() == content[:5]


@pytest.mark.parametrize("status,headers", [(206, {"Content-Range": "bytes 1-5/6"}), (200, {"Content-Length": "999"}), (201, {})])
def test_unexpected_range_length_and_status_cannot_install(tmp_path, status, headers):
    expected = asset()
    with pytest.raises(ValueError):
        setup._ensure_asset(expected, tmp_path / expected.filename, opener=lambda *a, **k: Response(b"bad", status, headers))
    assert not (tmp_path / expected.filename).exists()


def test_metadata_must_match_revision_license_and_lfs_hashes():
    metadata = {"sha": setup.REVISION, "cardData": {"license": "apache-2.0"},
                "siblings": [{"rfilename": value.filename, "size": value.size_bytes,
                              "lfs": {"sha256": value.sha256, "size": value.size_bytes}}
                             for value in (setup.MODEL, setup.PROJECTORS["q8_0"])]}
    setup.verify_metadata(metadata, "q8_0")
    for field, value in (("sha", "main"), ("cardData", {"license": "unknown"}), ("siblings", [])):
        with pytest.raises(ValueError):
            setup.verify_metadata({**metadata, field: value}, "q8_0")


def test_verify_only_checks_all_assets_and_manifest_without_network(tmp_path, monkeypatch):
    assets = [setup.Asset(f"asset-{i}.bin", 4, hashlib.sha256(b"data").hexdigest(), "https://example.invalid", role)
              for i, role in enumerate(("model", "projector", "model_card", "license"))]
    monkeypatch.setattr(setup, "MODEL", assets[0])
    monkeypatch.setattr(setup, "PROJECTORS", {"q8_0": assets[1]})
    monkeypatch.setattr(setup, "MODEL_CARD", assets[2])
    monkeypatch.setattr(setup, "LICENSE", assets[3])
    monkeypatch.setattr(setup.urllib.request, "urlopen", lambda *a, **k: pytest.fail("Verification must stay offline"))
    for expected in assets:
        (tmp_path / expected.filename).write_bytes(b"data")
    (tmp_path / "manifest.json").write_text(json.dumps(setup.build_manifest("q8_0")), encoding="utf8")
    assert setup.setup_assets(tmp_path, verify_only=True)["status"] == "verified"
    (tmp_path / assets[1].filename).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        setup.setup_assets(tmp_path, verify_only=True)


def test_parallel_ranges_reuse_prefix_and_assemble_only_verified_complete_file(tmp_path):
    content = b"a known model stream with multiple chunks"
    expected = asset(content)
    final = tmp_path / expected.filename
    (tmp_path / (expected.filename + ".part")).write_bytes(content[:4])
    requests = []
    def opener(request, timeout):
        extent = request.get_header("Range").removeprefix("bytes=")
        start, end = map(int, extent.split("-"))
        requests.append((start, end))
        assert start >= 4 and not final.exists()
        return Response(content[start:end + 1], 206, {"Content-Range": f"bytes {start}-{end}/{len(content)}",
                                                     "Content-Length": str(end - start + 1)})
    assert setup._ensure_parallel_asset(expected, final, 4, opener=opener, chunk_size=7)
    assert final.read_bytes() == content
    assert sum(end - start + 1 for start, end in requests) == len(content) - 4
    assert list(tmp_path.iterdir()) == [final]


def test_parallel_download_rejects_ignored_ranges_and_retains_original(tmp_path):
    content = b"known synthetic model"
    expected = asset(content)
    final = tmp_path / expected.filename
    final.write_bytes(b"previous verified artifact")
    with pytest.raises(ValueError, match="exact requested byte range"):
        setup._ensure_parallel_asset(expected, final, 4, opener=lambda *a, **k: Response(content), chunk_size=5)
    assert final.read_bytes() == b"previous verified artifact"


def test_parallel_full_hash_failure_discards_bad_parts_without_publishing(tmp_path):
    content = b"known synthetic model"
    expected = asset(content)
    final = tmp_path / expected.filename
    def opener(request, timeout):
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        return Response(b"x" * (end - start + 1), 206, {"Content-Range": f"bytes {start}-{end}/{len(content)}",
                                                       "Content-Length": str(end - start + 1)})
    with pytest.raises(ValueError, match="SHA-256"):
        setup._ensure_parallel_asset(expected, final, 4, opener=opener, chunk_size=5)
    assert not final.exists() and not list(tmp_path.iterdir())
