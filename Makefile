DOCTEST_FILES=bigraph_schema/type_system.py
TESTS=bigraph_schema/tests.py
LIBRARY=bigraph_schema/library.py
PY=PYTHONPATH=`pwd` uv run


tests: pytest doctest library

pytest:
	${PY} pytest

library:
	${PY} ${LIBRARY}

doctest:
	${PY} -m doctest ${DOCTEST_FILES}

debug:
	${PY} python3 -i ${TESTS}

debug-lib:
	${PY} python3 -i ${LIBRARY}
