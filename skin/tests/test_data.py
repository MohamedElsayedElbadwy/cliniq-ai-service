"""Leakage prevention tests on synthetic image manifests, not the real data."""
from collections import Counter

from skin.data.inspect_dataset import assign_splits, find_clusters


def test_cluster_transitive_duplicates_and_conflicts():
    rows = [
        dict(path='a', label='A', sha256='a', pixel_sha256='same', phash=5, dhash=5),
        dict(path='b', label='A', sha256='b', pixel_sha256='same', phash=5, dhash=5),
        dict(path='c', label='B', sha256='c', pixel_sha256='other', phash=5, dhash=5),
        dict(path='d', label='B', sha256='d', pixel_sha256='d', phash=2**64-1, dhash=2**64-1),
    ]
    groups = find_clusters(rows)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_stratified_group_split_deterministic_and_disjoint():
    rows = [dict(path=f'{label}/{i}', label=label, cluster_id=f'{label}-{i//2}')
            for label in ['A', 'B'] for i in range(100)]
    split = assign_splits(rows, seed=42)
    assert split == assign_splits(rows, seed=42)
    groups = {}
    for row, name in zip(rows, split):
        assert groups.setdefault(row['cluster_id'], name) == name
    for label in ['A', 'B']:
        counts = Counter(name for row, name in zip(rows, split) if row['label'] == label)
        assert counts == {'train':70, 'val':16, 'test':14} or counts == {'train':70, 'val':14, 'test':16}
