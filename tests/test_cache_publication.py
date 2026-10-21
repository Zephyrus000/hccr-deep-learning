from __future__ import annotations

import ctypes
import errno
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hccr.tda.features import CacheWriter, load_cache, save_cache


class CachePublicationTests(unittest.TestCase):
    def test_transient_windows_denial_does_not_discard_complete_cache(self):
        rename = Path.rename
        for streamed in (False, True):
            with self.subTest(streamed=streamed), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "published"
                failures = [5, 32]

                def locked_once(source, target, failures=failures, path=path):
                    if source.name == "cache" and failures:
                        error = PermissionError(errno.EACCES, "temporary Windows lock")
                        error.winerror = failures.pop(0)
                        self.assertFalse(path.exists())
                        raise error
                    return rename(source, target)

                with patch.object(Path, "rename", locked_once), patch("time.sleep"):
                    if streamed:
                        with CacheWriter(path, {"fixture": 1}, 2) as writer:
                            writer.append({"cnn": np.array([[1, 2], [3, 4]])})
                    else:
                        save_cache(
                            path, {"cnn": np.array([[1, 2], [3, 4]])}, {"fixture": 1}
                        )
                with load_cache(path, {"fixture": 1}) as cache:
                    np.testing.assert_array_equal(cache["cnn"], [[1, 2], [3, 4]])

    @unittest.skipUnless(os.name == "nt", "Windows directory sharing semantics")
    def test_real_windows_directory_handle_is_released_before_retry(self):
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.restype = wintypes.BOOL
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "published"
            writer = CacheWriter(path, {}, 2)
            writer.__enter__()
            writer.append({"cnn": np.array([[5], [6]])})
            handle = kernel.CreateFileW(
                str(writer.staged / "cnn.npy"), 0x80000000, 3, None, 3, 0, None
            )
            if handle == ctypes.c_void_p(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            held = True
            observed_errors = []
            rename = Path.rename

            def release_on_denial(source, target):
                nonlocal held
                try:
                    return rename(source, target)
                except PermissionError as error:
                    observed_errors.append(error.winerror)
                    kernel.CloseHandle(handle)
                    held = False
                    raise

            try:
                with patch.object(Path, "rename", release_on_denial):
                    writer.__exit__(None, None, None)
            finally:
                if held:
                    kernel.CloseHandle(handle)
            self.assertEqual(len(observed_errors), 1)
            self.assertIn(observed_errors[0], (5, 32))
            with load_cache(path, {}) as cache:
                np.testing.assert_array_equal(cache["cnn"], [[5], [6]])

    def test_persistent_denial_keeps_staging_and_refuses_overwrite(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "published"
            writer = CacheWriter(path, {}, 2)
            error = PermissionError(errno.EACCES, "persistent Windows denial")
            error.winerror = 5
            with (
                self.assertRaises(PermissionError),
                patch.object(Path, "rename", side_effect=error),
                patch("time.sleep"),
                writer,
            ):
                writer.append({"cnn": np.array([[7], [8]])})
            self.assertFalse(path.exists())
            with load_cache(writer.staged, {}) as cache:
                np.testing.assert_array_equal(cache["cnn"], [[7], [8]])
            save_cache(path, {"cnn": np.array([[99]])}, {})
            with self.assertRaises(FileExistsError), CacheWriter(path, {}, 1):
                pass
            with load_cache(path, {}) as cache:
                np.testing.assert_array_equal(cache["cnn"], [[99]])


if __name__ == "__main__":
    unittest.main()
