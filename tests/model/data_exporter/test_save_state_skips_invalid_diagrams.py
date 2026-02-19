import json


def test_save_state_loader_skips_invalid_diagrams(tmp_path, qapp):
    from nn_verification_visualisation.model.data_loader.save_state_loader import SaveStateLoader

    doc = {
        "format": "nnvv_save_state",
        "version": 1,
        "loaded_networks": [],
        "diagrams": [
            {"plot_generation_configs": [], "polygons": [], "plots": []},
        ],
    }

    f = tmp_path / "save_state.json"
    f.write_text(json.dumps(doc), encoding="utf-8")

    res = SaveStateLoader().load_save_state(str(f))
    assert res.is_success, res.error
    assert len(res.data.diagrams) == 0
