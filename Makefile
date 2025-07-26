DOCTEST_FILES=bigraph_schema/type_system.py

PYTEST_FILE=bigraph_schema/tests.py

tests: pytest doctest

pytest:
	uv run pytest

doctest:
	uv run python3 -m doctest ${DOCTEST_FILES} # -v

test-repl:
	PYTHONPATH=`pwd` uv run python3 -i ${PYTEST_FILE}
