.PHONY: clean test test-int

clean:
	rm -rf .cache
	rm -rf .pytest_cache

test:
	pytest
