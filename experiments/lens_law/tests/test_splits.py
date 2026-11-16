from lenslaw.splits import assign_split


def test_same_key_always_gets_the_same_split():
    assert assign_split("scene-001") == assign_split("scene-001")


def test_only_dev_and_test_are_produced():
    assert {assign_split(f"k{i}") for i in range(100)} == {"dev", "test"}


def test_dev_share_is_about_sixty_percent():
    n = 5000
    dev = sum(assign_split(f"k{i}") == "dev" for i in range(n))
    assert 0.57 <= dev / n <= 0.63


def test_changing_the_salt_changes_some_assignments():
    keys = [f"k{i}" for i in range(200)]
    assert [assign_split(k) for k in keys] != [assign_split(k, salt="other") for k in keys]
