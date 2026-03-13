from PySide6.QtCore import Qt, QModelIndex


def test_load_bounds_applies_in_range_skips_out_of_range_and_emits(qapp):
    from nn_verification_visualisation.model.data.input_bounds import InputBounds

    b = InputBounds(2)

    emissions = []
    b.dataChanged.connect(lambda tl, br, roles=None: emissions.append((tl, br, roles)))

    # key 5 is out-of-range -> should be skipped (continue branch)
    b.load_bounds({0: (0.0, 1.0), 5: (9.0, 9.0), 1: (-2.0, 3.0)})

    assert b.get_values() == [(0.0, 1.0), (-2.0, 3.0)]
    assert len(emissions) == 1

    tl, br, roles = emissions[0]
    assert tl.row() == 0 and tl.column() == 0
    assert br.row() == 1 and br.column() == 1
    assert Qt.ItemDataRole.DisplayRole in roles and Qt.ItemDataRole.EditRole in roles


def test_data_returns_none_for_invalid_index_or_role(qapp):
    from nn_verification_visualisation.model.data.input_bounds import InputBounds

    b = InputBounds(1)

    # invalid index -> None (covers line 100)
    assert b.data(QModelIndex(), role=Qt.ItemDataRole.DisplayRole) is None

    # wrong role -> None (covers role-not-accepted)
    assert b.data(b.index(0, 0), role=Qt.ItemDataRole.DecorationRole) is None


def test_setdata_branches_and_clamping_and_emits(qapp):
    from nn_verification_visualisation.model.data.input_bounds import InputBounds

    b = InputBounds(1)

    emissions = []
    b.dataChanged.connect(lambda tl, br, roles=None: emissions.append((tl, br, roles)))

    idx_lo = b.index(0, 0)
    idx_hi = b.index(0, 1)

    # wrong role -> False
    assert b.setData(idx_lo, 1.0, role=Qt.ItemDataRole.DisplayRole) is False

    # invalid index -> False
    assert b.setData(QModelIndex(), 1.0, role=Qt.ItemDataRole.EditRole) is False

    # value not float -> False (ValueError branch)
    assert b.setData(idx_lo, "not_a_number", role=Qt.ItemDataRole.EditRole) is False

    # set upper first to 5 -> ok, emits once
    assert b.setData(idx_hi, 5.0, role=Qt.ItemDataRole.EditRole) is True
    assert b.get_values()[0] == (0.0, 5.0)

    # set lower to 10 -> clamps down to upper (min branch col=0)
    assert b.setData(idx_lo, 10.0, role=Qt.ItemDataRole.EditRole) is True
    assert b.get_values()[0] == (5.0, 5.0)

    # set upper to -2 -> clamps up to lower (max branch col=1)
    assert b.setData(idx_hi, -2.0, role=Qt.ItemDataRole.EditRole) is True
    assert b.get_values()[0] == (5.0, 5.0)

    # only the 3 successful setData calls emit (each emits single-cell update)
    assert len(emissions) == 3
    for tl, br, roles in emissions:
        assert tl == br
        assert Qt.ItemDataRole.DisplayRole in roles and Qt.ItemDataRole.EditRole in roles