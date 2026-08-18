"""Dataset adapters, manifest contracts, and split audits."""

from hccr.data.dataset import HCCRDataset, select_class_subset
from hccr.data.folder_adapter import FolderSample, iter_folder_samples
from hccr.data.manifest import ManifestAudit, audit_manifest, read_manifest
from hccr.data.splitter import WriterDisjointSplitter

__all__ = [
    "FolderSample",
    "HCCRDataset",
    "ManifestAudit",
    "WriterDisjointSplitter",
    "audit_manifest",
    "iter_folder_samples",
    "read_manifest",
    "select_class_subset",
]
