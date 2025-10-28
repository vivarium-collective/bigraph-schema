DOCTEST_FILES=bigraph_schema/type_system.py
TESTS=bigraph_schema/tests.py
RUNTIME=bigraph_schema/runtime.py
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

debug-rt:
	${PY} python3 -i ${RUNTIME}
