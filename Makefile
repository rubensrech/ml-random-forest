all: run

run: 
	python3 lib/DecisionTree.py

clean:
	rm -f ./*/Digraph*
	rm -rf ./*/__pycache__
	rm -f Tree*.pdf
	rm -f Tree*.png