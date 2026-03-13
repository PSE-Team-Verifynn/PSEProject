from pathlib import Path

import pytest

class DummyObserver:
    def __init__(self):
        self.scheduled = []
        self.started = False
        self.stopped = False
        self.joined = False

    def schedule(self, handler, path, recursive=True):
        self.scheduled.append((handler, path, recursive))

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def join(self):
        self.joined = True

class DummyEvent:
    def __init__(self, src_path: str, is_directory: bool = False):
        self.src_path = src_path
        self.is_directory = is_directory


class DummyAlgo:
    def __init__(self, name="A", path="p.py", is_deterministic=True):
        self.name = name
        self.path = path
        self.is_deterministic = is_deterministic

class DummyStorage:
    def __init__(self):
        self.added = []
        self.removed = []
        self.modified = []

    def add_algorithm(self, algo):
        self.added.append(algo)

    def remove_algorithm(self, path):
        self.removed.append(path)

    def modify_algorithm(self, path, algo):
        self.modified.append((path, algo))


class DummyResult:
    def __init__(self, data=None, error=None):
        self.data = data
        self.error = error
        self.is_success = error is None


def _make_observer_instance(watch_dir: Path):
    """
    Create AlgorithmFileObserver instance WITHOUT running __init__
    (so we don't start watchdog threads).
    """
    from nn_verification_visualisation.model.data_loader.algorithm_file_observer import AlgorithmFileObserver

    obs = AlgorithmFileObserver.__new__(AlgorithmFileObserver)
    obs.watch_dir = watch_dir
    obs.ALLOWED_EXTENSIONS = (".py",)
    return obs


def test_initial_sync_adds_all_py_files(tmp_path, monkeypatch):
    """
    __initial_sync should add all *.py files under watch_dir recursively,
    ignore other files.
    """
    from nn_verification_visualisation.model.data_loader import algorithm_file_observer as mod

    # create fake algorithms dir structure
    alg_dir = tmp_path / "algorithms"
    alg_dir.mkdir()
    (alg_dir / "a.py").write_text("print('a')", encoding="utf-8")
    (alg_dir / "b.txt").write_text("ignore", encoding="utf-8")
    sub = alg_dir / "sub"
    sub.mkdir()
    (sub / "c.py").write_text("print('c')", encoding="utf-8")

    # patch Storage to capture add_algorithm calls
    class DummyStorage:
        def __init__(self):
            self.added = []

        def add_algorithm(self, algo):
            self.added.append(algo)

    storage = DummyStorage()
    monkeypatch.setattr(mod, "Storage", lambda: storage, raising=True)

    # patch AlgorithmLoader.load_algorithm to always succeed
    def fake_load_algorithm(path: str):
        return DummyResult(data=DummyAlgo(name=Path(path).stem, path=path))

    monkeypatch.setattr(mod.AlgorithmLoader, "load_algorithm", staticmethod(fake_load_algorithm), raising=True)

    obs = _make_observer_instance(alg_dir)
    obs._AlgorithmFileObserver__initial_sync()

    # should add only a.py and c.py
    assert [Path(a.path).name for a in storage.added] == ["a.py", "c.py"]
    assert [a.name for a in storage.added] == ["a", "c"]


def test_process_event_deleted_calls_remove(monkeypatch, tmp_path):
    from nn_verification_visualisation.model.data_loader import algorithm_file_observer as mod

    class DummyStorage:
        def __init__(self):
            self.removed = []

        def remove_algorithm(self, algo_path: str):
            self.removed.append(algo_path)

        def add_algorithm(self, _):  # not used
            raise AssertionError("add_algorithm should not be called for deleted")

        def modify_algorithm(self, *_):  # not used
            raise AssertionError("modify_algorithm should not be called for deleted")

    storage = DummyStorage()
    monkeypatch.setattr(mod, "Storage", lambda: storage, raising=True)

    obs = _make_observer_instance(tmp_path)
    e = DummyEvent(str(tmp_path / "x.py"), is_directory=False)

    obs._AlgorithmFileObserver__process_event(e, "deleted")

    assert storage.removed == [str(tmp_path / "x.py")]


def test_process_event_created_calls_add(monkeypatch, tmp_path):
    from nn_verification_visualisation.model.data_loader import algorithm_file_observer as mod

    class DummyStorage:
        def __init__(self):
            self.added = []

        def add_algorithm(self, algo):
            self.added.append(algo)

        def remove_algorithm(self, *_):
            raise AssertionError("remove_algorithm should not be called for created")

        def modify_algorithm(self, *_):
            raise AssertionError("modify_algorithm should not be called for created")

    storage = DummyStorage()
    monkeypatch.setattr(mod, "Storage", lambda: storage, raising=True)

    algo_file = tmp_path / "new_algo.py"

    monkeypatch.setattr(
        mod.AlgorithmLoader,
        "load_algorithm",
        staticmethod(lambda path: DummyResult(data=DummyAlgo(name="NEW", path=path))),
        raising=True,
    )

    obs = _make_observer_instance(tmp_path)
    e = DummyEvent(str(algo_file), is_directory=False)

    obs._AlgorithmFileObserver__process_event(e, "created")

    assert len(storage.added) == 1
    assert storage.added[0].name == "NEW"
    assert storage.added[0].path == str(algo_file)


def test_process_event_modified_calls_modify(monkeypatch, tmp_path):
    from nn_verification_visualisation.model.data_loader import algorithm_file_observer as mod

    class DummyStorage:
        def __init__(self):
            self.modified = []

        def modify_algorithm(self, algo_path: str, algo):
            self.modified.append((algo_path, algo))

        def add_algorithm(self, *_):
            raise AssertionError("add_algorithm should not be called for modified")

        def remove_algorithm(self, *_):
            raise AssertionError("remove_algorithm should not be called for modified")

    storage = DummyStorage()
    monkeypatch.setattr(mod, "Storage", lambda: storage, raising=True)

    algo_file = tmp_path / "algo.py"

    monkeypatch.setattr(
        mod.AlgorithmLoader,
        "load_algorithm",
        staticmethod(lambda path: DummyResult(data=DummyAlgo(name="M", path=path))),
        raising=True,
    )

    obs = _make_observer_instance(tmp_path)
    e = DummyEvent(str(algo_file), is_directory=False)

    obs._AlgorithmFileObserver__process_event(e, "modified")

    assert len(storage.modified) == 1
    path, algo = storage.modified[0]
    assert path == str(algo_file)
    assert algo.name == "M"


def test_process_event_ignores_non_py_and_directories(monkeypatch, tmp_path):
    from nn_verification_visualisation.model.data_loader import algorithm_file_observer as mod

    class DummyStorage:
        def __init__(self):
            self.calls = 0

        def add_algorithm(self, *_): self.calls += 1
        def modify_algorithm(self, *_): self.calls += 1
        def remove_algorithm(self, *_): self.calls += 1

    storage = DummyStorage()
    monkeypatch.setattr(mod, "Storage", lambda: storage, raising=True)

    # if loader were called, we'd see it; but it should be ignored
    monkeypatch.setattr(
        mod.AlgorithmLoader,
        "load_algorithm",
        staticmethod(lambda path: (_ for _ in ()).throw(AssertionError("Should not load"))),
        raising=True,
    )

    obs = _make_observer_instance(tmp_path)

    obs._AlgorithmFileObserver__process_event(DummyEvent(str(tmp_path / "x.txt"), False), "created")
    obs._AlgorithmFileObserver__process_event(DummyEvent(str(tmp_path / "dir"), True), "created")

    assert storage.calls == 0


def test_process_event_load_error_does_not_add_or_modify(monkeypatch, tmp_path):
    from nn_verification_visualisation.model.data_loader import algorithm_file_observer as mod

    class DummyStorage:
        def __init__(self):
            self.added = 0
            self.modified = 0

        def add_algorithm(self, *_): self.added += 1
        def modify_algorithm(self, *_): self.modified += 1
        def remove_algorithm(self, *_): pass

    storage = DummyStorage()
    monkeypatch.setattr(mod, "Storage", lambda: storage, raising=True)

    monkeypatch.setattr(
        mod.AlgorithmLoader,
        "load_algorithm",
        staticmethod(lambda path: DummyResult(data=None, error=RuntimeError("bad algo"))),
        raising=True,
    )

    obs = _make_observer_instance(tmp_path)
    e = DummyEvent(str(tmp_path / "bad.py"), is_directory=False)

    obs._AlgorithmFileObserver__process_event(e, "created")
    obs._AlgorithmFileObserver__process_event(e, "modified")

    assert storage.added == 0
    assert storage.modified == 0

def test_observer_init_returns_early_when_algorithms_dir_missing(monkeypatch, tmp_path):
    """
    Covers __init__ lines 20-33: if algorithms dir missing -> early return.
    """
    import nn_verification_visualisation.model.data_loader.algorithm_file_observer as mod

    # make Path(__file__).parents[4] point to tmp_path
    monkeypatch.setattr(mod, "__file__", str(tmp_path / "x" / "y" / "z" / "w" / "algorithm_file_observer.py"), raising=False)

    # ensure algorithms dir does NOT exist
    # (tmp_path exists, but tmp_path/algorithms not)
    monkeypatch.setattr(mod, "Observer", DummyObserver, raising=True)

    # avoid touching real Storage
    storage = DummyStorage()
    monkeypatch.setattr(mod, "Storage", lambda: storage, raising=True)

    # init should not crash even if it returns early
    obs = mod.AlgorithmFileObserver()
    # early return means observer probably not set
    assert getattr(obs, "observer", None) is None or isinstance(getattr(obs, "observer", None), DummyObserver) is False


def test_observer_init_sets_watch_dir_and_starts_when_dir_exists(monkeypatch, tmp_path):
    import nn_verification_visualisation.model.data_loader.algorithm_file_observer as mod
    from pathlib import Path

    # Make __file__ point somewhere deep so parents[4] resolves under tmp_path/"a"
    fake_file = tmp_path / "a" / "b" / "c" / "d" / "e" / "algorithm_file_observer.py"
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mod, "__file__", str(fake_file), raising=False)

    # Create the algorithms directory EXACTLY where the code will look:
    # parents[4] from .../a/b/c/d/e/file.py is .../a
    project_root = fake_file.parents[4]          # -> tmp_path / "a"
    alg_dir = project_root / "algorithms"
    alg_dir.mkdir(parents=True, exist_ok=True)
    (alg_dir / "a.py").write_text("print('a')", encoding="utf-8")

    # Patch Storage + Loader + Observer
    storage = DummyStorage()
    monkeypatch.setattr(mod, "Storage", lambda: storage, raising=True)
    monkeypatch.setattr(mod, "Observer", DummyObserver, raising=True)
    monkeypatch.setattr(mod.AlgorithmLoader, "load_algorithm",
                        staticmethod(lambda p: DummyResult(DummyAlgo("A", p))), raising=True)

    obs = mod.AlgorithmFileObserver()

    # Robust assertions (no brittle absolute path expectations)
    watch_dir = Path(obs.watch_dir)
    assert watch_dir.exists()
    assert watch_dir.name == "algorithms"
    assert (watch_dir / "a.py").exists()

    assert isinstance(obs.observer, DummyObserver)
    assert obs.observer.started is True
    assert obs.observer.scheduled
    assert Path(obs.observer.scheduled[0][1]) == watch_dir


def test_initial_sync_stops_on_first_load_error(monkeypatch, tmp_path):
    """
    Covers __initial_sync lines 83-90 (error branch returns early).
    """
    import nn_verification_visualisation.model.data_loader.algorithm_file_observer as mod

    alg_dir = tmp_path / "algorithms"
    alg_dir.mkdir()
    (alg_dir / "a.py").write_text("print('a')", encoding="utf-8")
    (alg_dir / "b.py").write_text("print('b')", encoding="utf-8")

    storage = DummyStorage()
    monkeypatch.setattr(mod, "Storage", lambda: storage, raising=True)

    calls = {"n": 0}

    def fake_load(p):
        calls["n"] += 1
        if calls["n"] == 1:
            return DummyResult(error=RuntimeError("bad algo"))
        return DummyResult(DummyAlgo("B", p))

    monkeypatch.setattr(mod.AlgorithmLoader, "load_algorithm", staticmethod(fake_load), raising=True)

    # create instance WITHOUT __init__ to call private method directly
    obs = mod.AlgorithmFileObserver.__new__(mod.AlgorithmFileObserver)
    obs.watch_dir = alg_dir
    obs.ALLOWED_EXTENSIONS = (".py",)

    obs._AlgorithmFileObserver__initial_sync()

    # because first fails -> should add nothing
    assert storage.added == []


def test_process_event_ignores_directory_and_non_py(monkeypatch, tmp_path):
    """
    Covers __process_event lines 46-50.
    """
    import nn_verification_visualisation.model.data_loader.algorithm_file_observer as mod

    storage = DummyStorage()
    monkeypatch.setattr(mod, "Storage", lambda: storage, raising=True)

    # loader must not be called
    monkeypatch.setattr(mod.AlgorithmLoader, "load_algorithm", staticmethod(lambda p: (_ for _ in ()).throw(AssertionError("should not load"))), raising=True)

    obs = mod.AlgorithmFileObserver.__new__(mod.AlgorithmFileObserver)
    obs.ALLOWED_EXTENSIONS = (".py",)

    obs._AlgorithmFileObserver__process_event(DummyEvent(str(tmp_path / "dir"), True), "created")
    obs._AlgorithmFileObserver__process_event(DummyEvent(str(tmp_path / "x.txt"), False), "created")

    assert storage.added == []
    assert storage.removed == []
    assert storage.modified == []


def test_stop_calls_observer_stop_and_join(monkeypatch, tmp_path):
    """
    Covers stop() lines 107-109.
    """
    import nn_verification_visualisation.model.data_loader.algorithm_file_observer as mod

    obs = mod.AlgorithmFileObserver.__new__(mod.AlgorithmFileObserver)
    obs.observer = DummyObserver()

    obs.stop()

    assert obs.observer.stopped is True
    assert obs.observer.joined is True