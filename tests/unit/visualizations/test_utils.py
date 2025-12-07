from pathlib import Path
import logging
import os

import pytest
from pytest_mock import mocker
import matplotlib.pyplot as plt

from explorica.visualizations import _utils

# constants

PLOT = plt.subplots(figsize=(10, 6))
plt.plot(ax=PLOT[1])

# tests for visualizations._utils.save_plot()

def test_save_plot_invalid_file_format():
    with pytest.raises(ValueError, match="Unsupported file format"):
        _utils.save_plot(PLOT[0], directory="./plot.mp4")


@pytest.mark.parametrize("directory, plot_name, expected",
[
    ("", "plot", "plot.png"),
    ("", "histplot", "histplot.png"),
    (".", "plot", "plot.png"),
    ("plots", "plot", "plots/plot.png"),
    ("plots/deep/nested", "chart", "plots/deep/nested/chart.png"),
    ("plots/", "diagram", "plots/diagram.png"),
    ("output", "my-plot_2024", "output/my-plot_2024.png"),
    ("output", "plot with spaces", "output/plot with spaces.png"),
    ("", "special_plot", "special_plot.png"),
])
def test_save_plot_without_filename(directory, plot_name, expected, tmp_path: Path):
    _utils.save_plot(PLOT[0], directory=str(tmp_path/directory), plot_name=plot_name)
    assert (tmp_path / expected).exists()

def test_save_plot_non_existent_directory(tmp_path: Path, caplog):
    # Checks save to exsistent directory
    existent_dir = tmp_path / "plot.png"
    
    with caplog.at_level(logging.WARNING):
        _utils.save_plot(PLOT[0], directory=str(existent_dir))
    assert existent_dir.exists()
    assert all("was created automatically." not in message for message in caplog.messages)
    
    # Checks save to non-exsistent directory
    non_existent_dir = tmp_path / "new" / "nested" / "directory"
    plot_path = non_existent_dir / "plot.png"
    
    with caplog.at_level(logging.WARNING):
        _utils.save_plot(PLOT[0], directory=str(plot_path))
    
    assert plot_path.exists()
    
    assert any("was created automatically." in message for message in caplog.messages)
    assert any(str(non_existent_dir) in message for message in caplog.messages)

@pytest.mark.parametrize("file_format, error",
[
    # Supported file-formats 
    ("eps", None), # Encapsulated Postscript,
    ("jpg", None), # Joint Photographic Experts Group
    ("jpeg", None), # Joint Photographic Experts Group
    ("pdf", None), # Portable Document Format
    ("png", None), # Portable Network Graphics
    ("ps", None), # Postscript
    ("raw", None), # Raw RGBA bitmap
    ("rgba", None), # Raw RGBA bitmap
    ("svg", None), # Scalable Vector Graphics
    ("svgz", None), # Scalable Vector Graphics
    ("tif", None), # Tagged Image File Format
    ("tiff", None), # Tagged Image File Format
    ("webp", None), # WebP Image Format

    # Unsupported file-formats
    ("mp4", ValueError),
    ("psd", ValueError),
    ("csv", ValueError),
    ("json", ValueError),
    ("gif", ValueError),
])
def test_save_plot_different_file_formats(file_format, error, tmp_path: Path):
    filename = "graph." + file_format
    if error is not None:
        with pytest.raises(error):
            _utils.save_plot(PLOT[0], directory=str(tmp_path/filename))
    else:
        _utils.save_plot(PLOT[0], directory=str(tmp_path/filename))
        assert (tmp_path/filename).exists()


@pytest.mark.parametrize("filename", ["test.PNG", "test.JPG", "test.Pdf", "test.SVG"])
def test_save_plot_case_insensitive_formats(filename, tmp_path: Path):
    """Checks that formats are case-insensitive"""
    plot_path = tmp_path / filename
    _utils.save_plot(PLOT[0], directory=str(plot_path))
    assert plot_path.exists()

def test_save_plot_permission_denied(tmp_path: Path):
    """Checks for PermissionError raise if save_plot does not have write permissions"""
    read_only_dir = tmp_path / "readonly"
    read_only_dir.mkdir()
    original_mode = read_only_dir.stat().st_mode

    _utils.save_plot(PLOT[0], directory=str(read_only_dir / "plot1.png"))
    assert (read_only_dir / "plot1.png").exists()

    read_only_dir.chmod(0o444)  # read only perms
    try:
        with pytest.raises(PermissionError):
            _utils.save_plot(PLOT[0], directory=str(read_only_dir / "plot2.png"))
    finally:
        read_only_dir.chmod(original_mode) # restoring default permissions
    

def test_save_plot_invalid_fig_object(tmp_path: Path):
    with pytest.raises(TypeError):
        _utils.save_plot(fig="not a figure", directory=str(tmp_path / "plot.png"))
    
    with pytest.raises(TypeError):
        _utils.save_plot(fig=None, directory=str(tmp_path / "plot.png"))
    
    fig = plt.figure()
    fig.canvas = None
    with pytest.raises(AttributeError):
        _utils.save_plot(fig, directory=str(tmp_path / "plot.png"))

@pytest.mark.parametrize("directory, expected_path", [
    ("file with spaces.png", "file with spaces.png"),
    ("dir with spaces/file.png", "dir with spaces/file.png"),

    ("файл.png", "файл.png"),
    ("目录/图.png", "目录/图.png"),
    ("café.png", "café.png"),

    ("file-with-dashes.png", "file-with-dashes.png"),
    ("file_with_underscores.png", "file_with_underscores.png"),
    ("file.with.dots.png", "file.with.dots.png"),
    ("file(mixed).png", "file(mixed).png"),
    ("file%name.png", "file%name.png"),
    ("file@name.png", "file@name.png"),
    
    ("level1/level2/file.png", "level1/level2/file.png"),
    ("a/b/c/d/e/f/file.png", "a/b/c/d/e/f/file.png"),

    ("dir with spaces/file with spaces.png", "dir with spaces/file with spaces.png"),
    ("mixed-dir/mixed_file.png", "mixed-dir/mixed_file.png"),
    ("目录/文件 with spaces.png", "目录/文件 with spaces.png"),

    ("backslash_dir/plot.png", "backslash_dir/plot.png"),
])
def test_save_plot_path_with_special_chars(tmp_path: Path, directory: str, expected_path: str):
    """Test saving plots to paths with special characters and Unicode."""
    expected_full_path = tmp_path / expected_path
    
    _utils.save_plot(PLOT[0], directory=str(tmp_path/directory))
    
    assert expected_full_path.exists()

    saved_path = expected_full_path.resolve()
    assert saved_path == expected_full_path


def test_save_plot_verbose(caplog, tmp_path: Path):
    plot_name = "plot"
    # Test with verbose=True
    with caplog.at_level(logging.INFO):
        _utils.save_plot(PLOT[0], directory=str(tmp_path), verbose=True, plot_name=plot_name)
    assert any(f"'{plot_name}' saved to {tmp_path}" in message for message in caplog.messages)

    # Clear the log records
    caplog.records.clear()
    
    # Test with verbose=False  
    with caplog.at_level(logging.INFO):
        _utils.save_plot(PLOT[0], directory=str(tmp_path), verbose=False, plot_name=plot_name)
    assert all(f"'{plot_name}' saved to {tmp_path}" not in message for message in caplog.messages)

def test_save_plot_empty_plot_name(tmp_path):
    with pytest.raises(ValueError):
        _utils.save_plot(PLOT[0], str(tmp_path), plot_name = "")

def test_save_plot_overwrite_existing_file(tmp_path, caplog):
    PLOT[0].savefig(tmp_path/"plot_a.png")
    with pytest.raises(FileExistsError):
        _utils.save_plot(PLOT[0], directory=str(tmp_path/"plot_a.png"), overwrite=False)
    assert any("Attempted to save plot to existing path" in message for message in caplog.messages)
    caplog.records.clear()
    _utils.save_plot(PLOT[0], directory=str(tmp_path/"plot_b.png"),  overwrite=False)
    assert not any("Attempted to save plot to existing path" in message for message in caplog.messages)

def test_save_plot_empty_plot_name(tmp_path):
    with pytest.raises(ValueError):
        _utils.save_plot(PLOT[0], directory=str(tmp_path/"plot.png"), plot_name="")
    with pytest.raises(ValueError):
        _utils.save_plot(PLOT[0], directory=str(tmp_path), plot_name="")

@pytest.mark.parametrize(
    "invalid_directory, expected_exception",
    [
        (None, TypeError), 
        ("", ValueError),
        ("  ", ValueError),
        (12345, TypeError),
        ([], TypeError),
    ],
    ids=[
        "None",
        "Empty String",
        "Whitespace Only",
        "Integer Type",
        "List Type",
    ]
)
def test_save_plot_invalid_directory(invalid_directory, expected_exception, mocker):
    mocker.patch.object(PLOT[0], 'savefig', autospec=True)
    mocker.patch('pathlib.Path.mkdir', autospec=True)
    if expected_exception is None:
        _utils.save_plot(PLOT[0], directory=invalid_directory)
    else:
        with pytest.raises(expected_exception):
            _utils.save_plot(PLOT[0], directory=invalid_directory)
