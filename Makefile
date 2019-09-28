all: run

run: 
	python3 lib/DecisionTree.py

clean:
	rm -f ./*/Digraph*
	rm -rf ./*/__pycache__