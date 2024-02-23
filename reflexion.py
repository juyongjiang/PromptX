from generators import model_factory, generator_factory
from executors import executor_factory
from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from typing import List


def run_reflexion(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    is_leetcode: bool = False
) -> None:
    gen = generator_factory(language) # create language specific generator
    model = model_factory(model_name) # create language specific model
    exe = executor_factory(language, is_leet=is_leetcode) # create langauge specific executor
    
    print_v = make_printv(verbose)

    num_items = len(dataset) # [{}, ...]
    num_success = resume_success_count(log_path) # original implementation is wrong
    for i, item in enumerate_resume(dataset, log_path):
        cur_pass = 0
        is_solved = False
        cur_func_impl = None
        
        # important components
        implementations = [] # generated solutions
        test_feedback = [] # exectuion feedback
        reflections = [] # questions + answers + execution feedback => self-reflection
        
        while cur_pass < pass_at_k and not is_solved:
            # generate internal unit tests
            if is_leetcode:
                tests_i = item['visible_tests']
            else:
                tests_i = gen.internal_tests(item["prompt"], model, 1) # the example unit tests in the prompt

            # first attempt
            #############Implementation#############
            cur_func_impl = gen.func_impl(item["prompt"], model, "simple") 
            implementations.append(cur_func_impl)
            assert isinstance(cur_func_impl, str)
            
            # check if all internal unit tests pass to get the feedback 
            #############Evaluation Feedback#############
            is_passing_internal, feedback, _ = exe.execute(cur_func_impl, tests_i) # feedback is the execution feedback of internal unit tests
            test_feedback.append(feedback) # only use internal unit tests to evaluate the implementation, it is not enough to get high-quality feedback
            # if solved, exit early
            if is_passing_internal:
                is_passing_unit_tests = exe.evaluate(item["entry_point"], cur_func_impl, item["test"], timeout=10)
                is_solved = is_passing_unit_tests
                num_success += int(is_passing_unit_tests) # num_success = num_success + 1 if is_passing else 0
                break # pass internal while not pass unit tests, then break. Is the internal info sufficient?
            
            """
            use self-reflection to iteratively improve
            """
            cur_iter = 1
            cur_feedback = feedback
            while cur_iter < max_iters:
                # get self-reflection
                #############Self-Reflection#############
                reflection = gen.self_reflection(cur_func_impl, cur_feedback, model) # no question needed (I think we can add prompt to make it imporved)
                reflections += [reflection]

                # # prompt + execution feedback
                # prompt = item["prompt"]
                # prompt += "\n[unit test results from previous impl]:\n"
                # prompt += cur_feedback

                # apply self-reflection in the next attempt
                # prompt + implementation + feedback + reflection => next implementation
                #############Optimized Implementation#############
                cur_func_impl = gen.func_impl(
                    func_sig=item["prompt"],
                    model=model,
                    strategy="reflexion",
                    prev_func_impl=cur_func_impl,
                    feedback=cur_feedback,
                    self_reflection=reflection,
                )
                implementations.append(cur_func_impl)
                assert isinstance(cur_func_impl, str)

                # check if all internal unit tests pass
                is_passing_internal, cur_feedback, _ = exe.execute(cur_func_impl, tests_i)
                test_feedback.append(cur_feedback)
                # if solved, check if it passes the real tests, exit early
                if is_passing_internal or cur_iter == max_iters - 1:
                    is_passing_unit_tests = exe.evaluate(item["entry_point"], cur_func_impl, item["test"], timeout=10)
                    if is_passing_unit_tests:
                        item["solution"] = cur_func_impl
                        is_solved = True
                        num_success += 1
                    break
                cur_iter += 1
            cur_pass += 1

        item["is_solved"] = is_solved
        item["reflections"] = reflections
        item["implementations"] = implementations
        item["test_feedback"] = test_feedback
        item["solution"] = cur_func_impl
        item["acc"] = round(num_success/(i+1), 2)
        write_jsonl(log_path, [item], append=True)
        print_v(f'completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 2)}')