import ast
import signal
import astunparse

from .utils import function_with_timeout, ExecuteResult, Executor
from typing import List


class PyExecutor(Executor):
    # is_passing_internal, cur_feedback, _ = exe.execute(cur_func_impl, tests_i)
    # test_i = ["assert add(1, 2) == 3", ...]
    def execute(self, func: str, tests: List[str], timeout: int = 5) -> ExecuteResult:
        # Combine function code and assert statement
        imports = 'from typing import *'
        func_test_list = [f'{imports}\n{func}\n{test}' for test in tests]

        # Run the tests and collect the results
        success_tests = []
        failed_tests = []
        is_passing = True
        num_tests = len(func_test_list)
        for i in range(num_tests):
            try:
                function_with_timeout(exec, (func_test_list[i], globals()), timeout)
                success_tests += [tests[i]]
            except Exception:
                output = get_output(func, tests[i], timeout=timeout)
                failed_tests += [f"{tests[i]} # output: {output}"]
                is_passing = False

        state = []
        for test in tests:
            if test in success_tests:
                state += [True]
            else:
                state += [False]
        state = tuple(state)

        feedback = "Tested passed:"
        for test in success_tests:
            feedback += f"\n{test}"
        feedback += "\n\nTests failed:"
        for test in failed_tests:
            feedback += f"\n{test}"
            
        return ExecuteResult(is_passing, feedback, state)

    # is_passing_unit_tests = exe.evaluate(item["entry_point"], cur_func_impl, item["test"], timeout=10)
    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
        """
        Evaluates the implementation on Human-Eval Python.
        probably should be written in a dataset-agnostic way but not now
        
        def check(candidate):
            assert candidate('') == 0
            assert candidate('x') == 1
            assert candidate('asdasnakj') == 9
        def test_check():
            check(strlen)
        test_check()\n
        """
        code = f"""{func}

{test}

check({name})
    """
        try:
            function_with_timeout(exec, (code, globals()), timeout)
            return True
        except Exception:
            return False

def get_output(func: str, assert_statement: str, timeout: int = 5) -> str:
    try:
        exec(f"from typing import *\n{func}", globals())
        func_call = get_call_str(assert_statement)
        output = function_with_timeout(eval, (func_call, globals()), timeout)
        return output
    except TimeoutError:
        return "TIMEOUT"
    except Exception as e:
        return str(e)
    
def get_call_str(assert_statement: str) -> str:
    ast_parsed = ast.parse(assert_statement)
    try:
        call_str = ast_parsed.body[0].test.left # type: ignore
    except:
        call_str = ast_parsed.body[0].test # type: ignore

    return astunparse.unparse(call_str).strip()


if __name__ == "__main__":
    # Test the function
    func = "def add(a, b):\n    while True:\n        x = 1\n    return a + b"
    tests = ["assert add(1, 2) == 3", "assert add(1, 2) == 4"]
    print(PyExecutor().execute(func, tests, timeout=1))
