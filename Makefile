DOCTEST_FILES=bigraph_schema/type_system.py
DEBUG_FILE=bigraph_schema/tests.py

tests: pytest doctest library

pytest:
	uv run pytest

library:
	PYTHONPATH=`pwd` uv run bigraph_schema/library.py

doctest:
	PYTHONPATH=`pwd` uv run python3 -m doctest ${DOCTEST_FILES}

debug:
	PYTHONPATH=`pwd` uv run python3 -i $(DEBUG_FILE)
