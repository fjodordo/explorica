import os
import warnings
import pytest
from pathlib import Path
from explorica._utils import convert_filepath, validate_path, enable_io_logs
import logging

logger = logging.getLogger("test_logger")


# -------------------------------
# Tests for convert_to_filepath
# -------------------------------

def test_convert_to_filepath_with_dir(tmp_path):
    tmp_dir = Path(tmp_path) / "folder"
    result = convert_filepath(tmp_dir, "default.pdf")
    assert result == tmp_dir / "default.pdf"

def test_convert_to_filepath_with_file(tmp_path):
    tmp_file = Path(tmp_path) / "file.pdf" 
    result = convert_filepath(tmp_file, "default.pdf")
    assert result == tmp_file

# -------------------------------
# Tests for validate_path
# -------------------------------

def test_validate_path_overwrite_error(tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("test")
    with pytest.raises(FileExistsError):
        validate_path(file_path, overwrite_check=True)

def test_validate_path_warns_if_dir_missing(tmp_path):
    non_existing_dir = tmp_path / "new_dir"
    with pytest.warns(UserWarning):
        validate_path(non_existing_dir, dir_exists_check=True)

def test_validate_path_permission_error(tmp_path):
    # make dir without write perms
    protected_dir = tmp_path / "protected"
    protected_dir.mkdir()
    os.chmod(protected_dir, 0o400)
    try:  # read-only
        with pytest.raises(PermissionError):
            validate_path(protected_dir, overwrite_check=False,
                          have_permissions_check=True)
    finally:
        # restore orignal perms
        os.chmod(protected_dir, 0o700)

# -------------------------------
# Tests for enable_io_logs
# -------------------------------

def test_enable_io_logs_logs_permission_error(caplog):
    @enable_io_logs()
    def raise_permission():
        raise PermissionError("nope")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(PermissionError):
            raise_permission()
    assert "Permission denied" in caplog.text

def test_enable_io_logs_logs_file_exists_error(caplog):
    @enable_io_logs()
    def raise_exists():
        raise FileExistsError("already there")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(FileExistsError):
            raise_exists()
    assert "File already exists" in caplog.text

def test_enable_io_logs_logs_generic_exception(caplog):
    @enable_io_logs()
    def raise_generic():
        raise ValueError("oops")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError):
            raise_generic()
    assert "Unexpected IO error" in caplog.text

def test_enable_io_logs_captures_and_reemits_directory_warning(caplog):
    logger = logging.getLogger(__name__)
    @enable_io_logs(logger)
    def fn():
        warnings.warn(
            "Directory 'output' does not exist. It will be created automatically.",
            UserWarning,
        )

    caplog.set_level(logging.WARNING)

    # warning must be visible
    with pytest.warns(UserWarning, match="Directory 'output' does not exist"):
        fn()

    # and warning must be logged
    assert any(
        "Captured warning in fn" in record.message
        and "Directory 'output' does not exist" in record.message
        for record in caplog.records
    )
