"""Retaining bounded diagnostic fields also bounds mapping traversal."""

from collections.abc import Mapping

from observability.log_records import safe_value


class CountedMapping(Mapping):
    def __init__(self):
        self.visited = 0

    def __len__(self):
        return 10_000

    def __iter__(self):
        for index in range(len(self)):
            self.visited += 1
            yield f'field_{index}'

    def __getitem__(self, key):
        return key


def test_mapping_sanitization_only_visits_retained_entries():
    mapping = CountedMapping()
    result = safe_value('details', mapping)
    assert len(result) == 30
    assert mapping.visited == 30
    assert list(result) == [f'field_{index}' for index in range(30)]
