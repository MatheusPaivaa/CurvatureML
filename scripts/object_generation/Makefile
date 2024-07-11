# Define variables to facilitate maintenance
PYTHON = python3.8
PIP = pip3.8
REQUIREMENTS = ./requirements.txt
MAIN_SCRIPT = main.py

# Check which version of pip is available (pip or pip3)
ifeq (, $(shell which pip))
  PIP = pip3
endif

# Default command when only 'make' is executed
all: install run

# Installation of dependencies
install:
	sudo $(PIP) install -U pip
	sudo $(PIP) install -r $(REQUIREMENTS)

# Execution of the main script
run:
	$(PYTHON) $(MAIN_SCRIPT)

# Clean the project: remove .pyc files and __pycache__ directories
clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

# Command to help debug the Makefile (call with 'make print-VARIABLE')
print-%: ; @echo $*=$($*)
