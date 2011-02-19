# simple makefile to simplify repetetive build env management tasks under posix
# 	grabbed from scikit.learn

# for windows users: you need posix command line tools in PATH
#
PYTHON ?= python
NOSETESTS ?= nosetests
CTAGS ?= ctags

all: inplace test

clean:
	$(PYTHON) setup.py clean

in: inplace 
inplace:
	$(PYTHON) setup.py build_ext -i

test: in
	$(NOSETESTS) 

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) -R *
