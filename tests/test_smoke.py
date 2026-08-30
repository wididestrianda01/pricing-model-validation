def test_package_imports_and_exposes_version():
    import pricing

    assert isinstance(pricing.__version__, str)
    assert pricing.__version__
