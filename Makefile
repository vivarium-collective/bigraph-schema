DOCTEST_FILES=bigraph_schema/type_system.py
tests: pytest doctest

pytest:
	uv run pytest

doctest:
	exec uv run python3 -m doctest ${DOCTEST_FILES} # -v
