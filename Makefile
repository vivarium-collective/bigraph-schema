DOCTEST_FILES=bigraph_schema/type_system.py
TESTS=bigraph_schema/tests.py
RUNTIME=bigraph_schema/core.py
PY=PYTHONPATH=`pwd` uv run


tests: pytest doctest runtime

pytest:
	${PY} pytest

runtime:
	${PY} ${RUNTIME}

doctest:
	${PY} -m doctest ${DOCTEST_FILES}

debug:
	${PY} python3 -i ${TESTS}

core:
	${PY} python3 -i ${RUNTIME}

bigraph:
	${PY} python3 -i bigraph_schema/bigraph.py
