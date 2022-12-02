### Building and pushing to pip
python setup.py sdist
python setup.py bdist_wheel --universal
twine upload --repository-url https://test.pypi.org/legacy/ dist/cytocipher-0.1.3.tar.gz

### Downloading for test
## Instructions from: https://stackoverflow.com/questions/34514703/pip-install-from-pypi-works-but-from-testpypi-fails-cannot-find-requirements
#pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple cytocipher==0.1.3



