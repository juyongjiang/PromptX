from generators.model import ModelBase, Message
import random
import re
from typing import Union, List, Optional, Callable
from abc import abstractmethod, ABC


class Generator:
    @abstractmethod
    def self_reflection(self, func: str, feedback: str, model: ModelBase) -> str:
        ...

    @abstractmethod
    def func_impl(
        self,
        func_sig: str,
        model: ModelBase,
        strategy: str,
        prev_func_impl: Optional[str] = None,
        feedback: Optional[str] = None,
        self_reflection: Optional[str] = None,
        num_comps: int = 1,
        temperature: float = 0.0,
    ) -> Union[str, List[str]]:
        ...

    @abstractmethod
    def internal_tests(
            self,
            func_sig: str,
            model: ModelBase,
            max_num_tests: int = 5
    ) -> List[str]:
        ...


def generic_generate_func_impl(
    func_sig: str,
    model: ModelBase,
    strategy: str,
    prev_func_impl,
    feedback,
    self_reflection,
    num_comps,
    temperature,
    reflection_chat_instruction: str,
    reflection_few_shot: str,
    simple_chat_instruction: str,
    reflection_completion_instruction: str,
    simple_completion_instruction: str,
    code_block_instruction: str,
    parse_code_block: Callable[[str], str],
    add_code_block: Callable[[str], str]
) -> Union[str, List[str]]:
    if strategy != "reflexion" and strategy != "simple":
        raise ValueError(f"Invalid strategy: given `{strategy}` but expected one of `reflexion` or `simple`")
    if strategy == "reflexion" and (prev_func_impl is None or feedback is None or self_reflection is None):
        raise ValueError(f"Invalid arguments: given `strategy=reflexion` but `prev_func_impl`, `feedback`, or `self_reflection` is None")

    if model.is_chat:
        if strategy == "reflexion":
            # {reflection_few_shot}\n
            message = f"[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[unit test results from previous impl]:\n{feedback}\n\n[reflection on previous impl]:\n{self_reflection}\n\n[improved impl]:\n{func_sig}"
            prompt = f"{reflection_chat_instruction}\n{code_block_instruction}"
            # func_bodies is a really bad name, as it can also be just 1 string
            print_messages(prompt + "\n" + f"{reflection_few_shot}\n", message)
            messages = [
                Message(
                    role="system",
                    content=prompt,
                ),
                Message(
                    role="user", # TODO: check this
                    content=reflection_few_shot,
                ),
                Message(
                    role="assistant",
                    content=add_code_block(prev_func_impl),
                ),
                Message(
                    role="user",
                    content=f"[unit test results from previous impl]:\n{feedback}\n\n[reflection on previous impl]:",
                ),
                Message(
                    role="assistant",
                    content=self_reflection,
                ),
                Message(
                    role="user",
                    content=f"[improved impl]:\n{func_sig}",
                ),
            ]
            func_bodies = model.generate_chat(messages=messages, num_comps=num_comps, temperature=temperature)
        else:
            system_prompt = f"{simple_chat_instruction}\n{code_block_instruction}"
            print_messages(system_prompt, func_sig)
            messages = [
                Message(
                    role="system",
                    content=system_prompt,
                ),
                Message(
                    role="user",
                    content=func_sig,
                ),
            ]
            func_bodies = model.generate_chat(messages=messages, num_comps=num_comps, temperature=temperature)
    else:
        if strategy == "reflexion":
            prompt = f"{reflection_completion_instruction}\n{add_code_block(prev_func_impl)}\n\nunit tests:\n{feedback}\n\nhint:\n{self_reflection}\n\n# improved implementation\n{func_sig}\n{code_block_instruction}"
            func_bodies = model.generate(prompt, num_comps=num_comps, temperature=temperature)
        else:
            prompt = f"{simple_completion_instruction}\n{func_sig}\n{code_block_instruction}"
            func_bodies = model.generate(prompt, num_comps=num_comps, temperature=temperature)

    if num_comps == 1:
        assert isinstance(func_bodies, str)
        # parse code block from the generated response
        func_body_str = parse_code_block(func_bodies)
        print_generated_func_body(func_body_str)
        return func_body_str

    else:
        func_bodies = [parse_code_block(func_body) for func_body in func_bodies]
        print_generated_func_body("\n\n".join(func_bodies))
        return func_bodies


def generate_with_accumulated_context(
    func_sig: str,
    model: ModelBase,
    strategy: str,
    prev_func_impl,
    accumulated_feedback,
    accumulated_reflection,
    num_comps,
    temperature,
    reflection_chat_instruction: str,
    reflection_few_shot: str,
    simple_chat_instruction: str,
    reflection_completion_instruction: str,
    simple_completion_instruction: str,
    code_block_instruction: str,
    parse_code_block: Callable[[str], str],
    add_code_block: Callable[[str], str]
) -> Union[str, List[str]]:
    # Ensure that the strategy is valid
    if strategy != "reflexion" and strategy != "simple":
        raise ValueError(
            f"Invalid strategy: given `{strategy}` but expected one of `reflexion` or `simple`")
    if strategy == "reflexion" and (prev_func_impl is None or accumulated_feedback is None or accumulated_reflection is None):
        raise ValueError(
            f"Invalid arguments: given `strategy=reflexion` but `prev_func_impl`, `feedback`, or `self_reflection` is None")

    # Build the accumulated context from the provided feedback and reflections
    accumulated_context = "\n\n".join(
        [f"[previous impl {i+1}]:\n{add_code_block(impl)}\n[unit test results from previous impl {i+1}]:\n{feedback}\n[reflection on previous impl {i+1}]:\n{reflection}" 
         for i, (impl, feedback, reflection) in enumerate(zip(prev_func_impl, accumulated_feedback, accumulated_reflection))]
    )

    if model.is_chat:
        if strategy == "reflexion":
            # Constructing the message using a loop for accumulated context
            messages = [
                Message(role="system", content=f"{reflection_chat_instruction}\n{code_block_instruction}"),
                Message(role="user", content=reflection_few_shot)
            ]
            
            for impl, feedback, reflection in zip(prev_func_impl, accumulated_feedback, accumulated_reflection):
                messages.append(Message(role="assistant", content=add_code_block(impl)))
                messages.append(Message(role="user", content=f"[unit test results from previous impl]:\n{feedback}\n\n[reflection on previous impl]:\n{reflection}"))
            
            messages.append(Message(role="user", content=f"[improved impl]:\n{func_sig}"))
            prompt = "\n".join([message.content for message in messages])
            message = (f"{reflection_few_shot}\n{accumulated_context}\n\n[improved impl]:\n{func_sig}")
            print_messages(prompt, message)

            func_bodies = model.generate_chat(messages=messages, num_comps=num_comps, temperature=temperature)
        else:
            system_prompt = f"{simple_chat_instruction}\n{code_block_instruction}"
            print_messages(system_prompt, func_sig)
            messages = [
                Message(role="system", content=f"{simple_chat_instruction}\n{code_block_instruction}"),
                Message(role="user", content=func_sig)
            ]
            func_bodies = model.generate_chat(messages=messages, num_comps=num_comps, temperature=temperature)
    else:
        if strategy == "reflexion":
            prompt = f"{reflection_completion_instruction}\n{accumulated_context}\n\n# improved implementation\n{func_sig}\n{code_block_instruction}"
            func_bodies = model.generate(prompt, num_comps=num_comps, temperature=temperature)
            print_messages(prompt, "")  
        else:
            prompt = f"{simple_completion_instruction}\n{func_sig}\n{code_block_instruction}"
            func_bodies = model.generate(prompt, num_comps=num_comps, temperature=temperature)
            print_messages(prompt, "")

    if num_comps == 1:
        assert isinstance(func_bodies, str)
        func_body_str = parse_code_block(func_bodies)
        print_generated_func_body(func_body_str)
        return func_body_str

    else:
        func_bodies = [parse_code_block(func_body) for func_body in func_bodies]
        print_generated_func_body("\n\n".join(func_bodies))
        return func_bodies
    

def print_generated_func_body(func_body_str: str) -> None:
    print(f"""--------------------- GENERATED FUNC BODY ---------------------
{func_body_str}
------------------------------------------""")

def generic_generate_internal_tests(
        func_sig: str,
        model: ModelBase,
        max_num_tests: int,
        test_generation_few_shot: str,
        test_generation_chat_instruction: str,
        test_generation_completion_instruction: str,
        parse_tests: Callable[[str], List[str]],
        is_syntax_valid: Callable[[str], bool],
        is_react: bool = False
) -> List[str]:
    """Generates tests for a function."""
    if model.is_chat:
        if is_react:
            messages = [
                Message(
                    role="system",
                    content=test_generation_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f"{test_generation_few_shot}\n\n[func signature]:\n{func_sig}\n\n[think]:"
                )
            ]
            output = model.generate_chat(messages=messages, max_tokens=1024)
            print(f'React test generation output: {output}')
        else:
            messages = [
                Message(
                    role="system",
                    content=f"{test_generation_chat_instruction}\n\n{test_generation_few_shot}",
                ),
                Message(
                    role="user",
                    content=f"[func signature]:\n{func_sig}\n\n[unit tests]:",
                )
            ]
            output = model.generate_chat(messages=messages, max_tokens=1024)
    else:
        prompt = f'{test_generation_completion_instruction}\n\nfunc signature:\n{func_sig}\nunit tests:'
        output = model.generate(prompt, max_tokens=1024)
    
    # extract the tests from the output
    all_tests = parse_tests(output)  # type: ignore
    # filter out invalid tests
    valid_tests = [test for test in all_tests if is_syntax_valid(test)]

    return sample_n_random(valid_tests, max_num_tests)

def sample_n_random(items: List[str], n: int) -> List[str]:
    """Sample min(n, len(items)) random items from a list"""
    assert n >= 0
    if n >= len(items):
        return items
    return random.sample(items, n)


def generic_generate_self_reflection(
        func: str,
        feedback: str,
        model: ModelBase,
        self_reflection_chat_instruction: str,
        self_reflection_completion_instruction: str,
        add_code_block: Callable[[str], str],
        self_reflection_few_shot: Optional[str] = None,
) -> str:
    if model.is_chat:
        if self_reflection_few_shot is not None:
            messages = [
                Message(
                    role="system",
                    content=self_reflection_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f'{self_reflection_few_shot}\n\n[function impl]:\n{add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\n[self-reflection]:',
                )
            ]
            reflection = model.generate_chat(messages=messages)
        else:
            messages = [
                Message(
                    role="system",
                    content=self_reflection_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f'[function impl]:\n{add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\n[self-reflection]:',
                )
            ]
            reflection = model.generate_chat(messages=messages)
    else:
        if self_reflection_few_shot is not None:
            reflection = model.generate(f'{self_reflection_completion_instruction}\n{self_reflection_few_shot}\n\n[function impl]:\n{add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\n[self-reflection]:')
        else:
            reflection = model.generate(f'{self_reflection_completion_instruction}\n{add_code_block(func)}\n\n{feedback}\n\nExplanation:') # I think something is missing
    return reflection  # type: ignore

def print_messages(system_message_text: str, user_message_text: str) -> None:
    print(f"""----------------------- SYSTEM MESSAGE -----------------------)
{system_message_text}
----------------------------------------------
----------------------- USER MESSAGE -----------------------
{user_message_text}
----------------------------------------------
""", flush=True)


# parse code block ("""code""") from generated response
def parse_code_block(string: str, lang: str) -> Optional[str]:
    code_pattern = fr"```{lang}\n(.*?)\n```"
    match = re.search(code_pattern, string, re.DOTALL)
    if match:
        return match.group(1)

    generic_code_pattern = r"```\n(.*?)\n```"
    match = re.search(generic_code_pattern, string, re.DOTALL)
    if match:
        return match.group(1)

    return parse_first_func(string, lang)


def parse_first_func(code: str, lang: str) -> Optional[str]:
    assert lang == "python", "Only python is supported for now. TODO: Rust"
    code_lines = code.split("\n")
    def_i = -1
    last_i = 0
    got_return = False
    for i, line in enumerate(code_lines):
        if line.startswith("def "):
            if def_i == -1:
                def_i = i
            else:
                break
        elif "return" in line and def_i != -1:
            got_return = True
        if line == "" and def_i != -1 and got_return:
            last_i = i
            break

    if last_i == 0:
        last_i = len(code_lines) - 1

    if def_i == -1:
        return None

    return "\n".join(code_lines[def_i:last_i+1]).rstrip("[/PYTHON]")


def add_code_block(string: str, lang: str) -> str:
    return f"```{lang}\n{string}\n```"



# fix indentation problem in python code
def py_fix_indentation(func_body: str) -> str:
    func_body = fix_turbo_response(func_body)
    """
    3 cases:
        1. good syntax
        2. first line not good
        3. entire body not good
    """
    def parse_indent_rec(f_body: str, cur_state: int) -> str:
        f_body = fix_markdown(f_body)
        if cur_state > 1:
            return f_body
        code = f'{DUMMY_FUNC_SIG}\n{f_body}\n{DUMMY_FUNC_CALL}'
        try:
            exec(code)
            return f_body
        except (IndentationError, SyntaxError):
            p_func = handle_first_line_indent if cur_state == 0 else handle_entire_body_indent
            return parse_indent_rec(p_func(func_body), cur_state + 1)
        except Exception:
            return f_body
    return parse_indent_rec(func_body, 0)


DUMMY_FUNC_SIG = "def func():"
DUMMY_FUNC_CALL = "func()"


def handle_first_line_indent(func_body: str) -> str:
    if func_body.startswith("    "):
        return func_body
    split = func_body.splitlines()
    return f"    {split[0]}\n" + "\n".join(split[1:])


def handle_entire_body_indent(func_body: str) -> str:
    split = func_body.splitlines()
    res = "\n".join(["    " + line for line in split])
    return res


def fix_markdown(func_body: str) -> str:
    return re.sub("`{3}", "", func_body)

def fix_turbo_response(func_body: str) -> str:
    return fix_markdown(remove_unindented_signatures(func_body))

def remove_unindented_signatures(code: str) -> str:
    regex = r"^def\s+\w+\s*\("

    before_signature = []
    after_signature = []
    signature_found = False

    for line in code.split("\n"):
        if re.match(regex, line):
            signature_found = True
            continue

        if signature_found:
            after_signature.append(line)
        else:
            if not line.startswith("    ") and line.strip():
                line = "    " + line
            before_signature.append(line)

    return "\n".join(before_signature + after_signature)


if __name__ == "__main__":
    CODE = """
aldaas
sub_parser = parser.add_subparsers().add_parser("frf
a")

def my_wonderful_func():
    def useless_helper():
        return 1
    if 1:
        return 1
    else:
        return (
            1,
            2,
        )

sadsadsa
2023-08-04dsa
dsa

def bleh():
    return aaa
"""
    print(parse_code_block(CODE, "python"))
    CODE = """def total_match(lst1: List[str], lst2: List[str]) -> List[str]:
    \"\"\"
    Write a function that accepts two lists of strings and returns the list that has
    total number of chars in the all strings of the list less than the other list.
    
    if the two lists have the same number of chars, return the first list.
    
    Examples
    >>> total_match([], [])
    []
    >>> total_match(['hi', 'admin'], ['hI', 'Hi'])
    ['hI', 'Hi']
    >>> total_match(['hi', 'admin'], ['hi', 'hi', 'admin', 'project'])
    ['hi', 'admin']
    >>> total_match(['hi', 'admin'], ['hI', 'hi', 'hi'])
    ['hI', 'hi', 'hi']
    >>> total_match(['4'], ['1', '2', '3', '4', '5'])
    ['4']
    \"\"\"
    total_chars_lst1 = sum(len(word) for word in lst1)
    total_chars_lst2 = sum(len(word) for word in lst2)
    
    if total_chars_lst1 < total_chars_lst2:
        return lst1
    elif total_chars_lst1 > total_chars_lst2:
        return lst2
    else:
        return lst1
    """
    print(parse_code_block(CODE, "python"))