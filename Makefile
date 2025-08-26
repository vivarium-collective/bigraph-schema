DOCTEST_FILES=bigraph_schema/type_system.py

tests: pytest doctest library

pytest:
	uv run pytest

library:
	uv run bigraph_schema/library.py

doctest:
	uv run python3 -m doctest ${DOCTEST_FILES} # -v
