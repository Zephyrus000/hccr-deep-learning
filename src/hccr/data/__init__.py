"""Dataset adapters, manifest contracts, and split audits."""

from hccr.data.dataset import HCCRDataset, resolve_data_root, select_class_subset
from hccr.data.folder_adapter import FolderSample, iter_folder_samples
from hccr.data.lmdb_store import LMDBImageStore, build_lmdb_image_store
from hccr.data.manifest import ManifestAudit, audit_manifest, read_manifest
from hccr.data.splitter import WriterDisjointSplitter

__all__ = [
    "FolderSample",
    "HCCRDataset",
    "LMDBImageStore",
    "ManifestAudit",
    "WriterDisjointSplitter",
    "audit_manifest",
    "build_lmdb_image_store",
    "iter_folder_samples",
    "read_manifest",
    "resolve_data_root",
    "select_class_subset",
]
