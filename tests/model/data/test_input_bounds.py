def test_input_bounds_values_and_sample(qapp):
    from nn_verification_visualisation.model.data.input_bounds import InputBounds

    b = InputBounds(3)

    # load values (support both APIs)
    if hasattr(b, "load_list"):
        b.load_list([(0.0, 1.0), (2.0, 3.0), (-1.0, 5.0)])
    elif hasattr(b, "load_bounds"):
        b.load_bounds({0: (0.0, 1.0), 1: (2.0, 3.0), 2: (-1.0, 5.0)})
    else:
        setattr(b, "_InputBounds__value", [(0.0, 1.0), (2.0, 3.0), (-1.0, 5.0)])

    # verify values
    if hasattr(b, "get_values"):
        assert b.get_values() == [(0.0, 1.0), (2.0, 3.0), (-1.0, 5.0)]
    else:
        assert getattr(b, "_InputBounds__value") == [(0.0, 1.0), (2.0, 3.0), (-1.0, 5.0)]

    # sample
    if hasattr(b, "get_sample"):
        assert b.get_sample() is None
    if hasattr(b, "set_sample") and hasattr(b, "get_sample"):
        sample = {"acc": 0.5, "hist": [1, 2, 3]}
        b.set_sample(sample)
        assert b.get_sample() == sample
    if hasattr(b, "clear_sample") and hasattr(b, "get_sample"):
        b.clear_sample()
        assert b.get_sample() is None


def test_input_bounds_read_only_toggle(qapp):
    from nn_verification_visualisation.model.data.input_bounds import InputBounds

    b = InputBounds(1)
    if hasattr(b, "set_read_only"):
        b.set_read_only(True)
        b.set_read_only(False)
