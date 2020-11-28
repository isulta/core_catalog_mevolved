generate: clean cc_generate.py
	python cc_generate.py

optimize: clean cc_optimize.py
	python cc_optimize.py

fitting: clean cc_generate.py
	python cc_generate_fitting.py
	
clean:
	rm -rf *.pyc __pycache__ /home/isultan/projects/itk/itk.pyc