.PHONY: build_dev_requirements install_dev_requirements install test

build_dev_requirements:
    pip-compile requirements.in

install_dev_requirements:
    pip install -r requirements.txt

install:
    python setup.py install

test:
	pytest tests