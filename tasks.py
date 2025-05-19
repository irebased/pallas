from invoke import task

@task
def test(ctx):
    """Run tests"""
    ctx.run("pytest")

@task
def test_coverage(ctx):
    """Run tests with coverage"""
    ctx.run("pytest --cov=pallas --cov-report=term-missing --cov-report=html")

@task
def clean(ctx):
    """Clean up build artifacts and cache"""
    ctx.run("rm -rf build/ dist/ *.egg-info .coverage htmlcov/ .pytest_cache/")