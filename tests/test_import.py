"""
Test that the package can be imported correctly.
"""

def test_import():
    """Test that the package can be imported."""
    try:
        from svg_mmnist_fd import frechet_distance
        assert callable(frechet_distance)
    except ImportError as e:
        assert False, f"Failed to import svg_mmnist_fd: {e}" 