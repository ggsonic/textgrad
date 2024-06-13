#simple coding test just for fun
import textgrad as tg
import random
import time

problem_text = """Longest Increasing Subsequence (LIS)

Problem Statement:
Given a sequence of integers, find the length of the longest subsequence that is strictly increasing. A subsequence is a sequence that can be derived from another sequence by deleting some or no elements without changing the order of the remaining elements.

Input:
The input consists of a list of integers representing the sequence.

Output:
The output should be an integer representing the length of the longest increasing subsequence."""

from IPython.core.interactiveshell import InteractiveShell

def run_function_in_interpreter(func_code):
    #raise Exception("This function will run the code returned by GPT-4o. Remove this if you'd like to run the code!")
    interpreter = InteractiveShell.instance()
    
    interpreter.run_cell(func_code, store_history=False, silent=True)
    
    func_name = func_code.split("def ")[1].split("(")[0].strip()
    func = interpreter.user_ns[func_name]
    
    return func


def test_longest_increasing_subsequence(fn):
    nums = [10, 22, 9, 33, 21, 50, 41, 60]
    assert fn(nums) == 5

    nums = [7, 2, 1, 3, 8, 4, 9, 6, 5]
    assert fn(nums) == 4

    nums = [5, 4, 3, 2, 1]
    assert fn(nums) == 1

    nums = [1, 2, 3, 4, 5]
    assert fn(nums) == 5

    nums = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    assert fn(nums) == 4

    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    assert fn(nums) == 4

    nums = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
    assert fn(nums) == 6

initial_solution = """
def longest_increasing_subsequence(nums):
    n = len(nums)
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    max_length = max(dp)
    lis = []
    
    for i in range(n - 1, -1, -1):
        if dp[i] == max_length:
            lis.append(nums[i])
            max_length -= 1
    
    return len(lis[::-1])
"""

longest_increasing_subsequence = run_function_in_interpreter(initial_solution)
test_longest_increasing_subsequence(longest_increasing_subsequence)

# Generate a random test case
def generate_random_test_case(size, min_value, max_value):
    return [random.randint(min_value, max_value) for _ in range(size)]

# Test the function with a random test case
size = 10000  # Adjust the size as needed
min_value = 1
max_value = 1000

nums = generate_random_test_case(size, min_value, max_value)
start_time = time.time()
lis = longest_increasing_subsequence(nums)
end_time = time.time()

print(f"Test Case Size: {size}")
print(f"Longest Increasing Subsequence Length: {lis}")
print(f"Runtime: {end_time - start_time:.5f} seconds")

base_url="http://xxxx:xxxx/v1"
api_key="xxxxx"
model_name="gpt-4o"
#tg.set_backward_engine(model_name,override=True,base_url=base_url,api_key=api_key)
llm_engine = tg.get_engine(model_name,base_url=base_url,api_key=api_key)
tg.set_backward_engine(llm_engine)


# Code is the variable of interest we want to optimize -- so requires_grad=True
code = tg.Variable(value=initial_solution,
                   requires_grad=True,
                   role_description="code instance to optimize")

# We are not interested in optimizing the problem -- so requires_grad=False
problem = tg.Variable(problem_text, 
                      requires_grad=False, 
                      role_description="the coding problem")

# Let TGD know to update code!
optimizer = tg.TGD(parameters=[code])

# The system prompt that will guide the behavior of the loss function.
loss_system_prompt = "You are a smart language model that evaluates code snippets. You do not solve problems or propose new code snippets, only evaluate existing solutions critically and give very concise feedback."
loss_system_prompt = tg.Variable(loss_system_prompt, requires_grad=False, role_description="system prompt to the loss function")

# The instruction that will be the prefix
instruction = """Think about the problem and the code snippet. Does the code solve the problem? What is the runtime complexity?"""

# The format string and setting up the call
format_string = "{instruction}\nProblem: {{problem}}\nCurrent Code: {{code}}"
format_string = format_string.format(instruction=instruction)

fields = {"problem": None, "code": None}
formatted_llm_call = tg.autograd.FormattedLLMCall( engine=llm_engine,
                                                  format_string=format_string,
                                                  fields=fields,
                                                  system_prompt=loss_system_prompt)

# Finally, the loss function
def loss_fn(problem: tg.Variable, code: tg.Variable) -> tg.Variable:
    inputs = {"problem": problem, "code": code}
    return formatted_llm_call(inputs=inputs, response_role_description=f"evaluation of the {code.get_role_description()}")

loss = loss_fn(problem, code)
print(loss.value)

print("xxxxxxx")
loss.backward()
print(code.gradients)

optimizer.step()

print("-------")
optimizer.zero_grad()
loss = loss_fn(problem, code)
loss.backward()
optimizer.step()
print(code.value)
print("-------")
optimizer.zero_grad()
loss = loss_fn(problem, code)
loss.backward()
optimizer.step()
print(code.value)
longest_increasing_subsequence = run_function_in_interpreter(code.value)

start_time = time.time()
lis = longest_increasing_subsequence(nums)
end_time = time.time()

print(f"Longest Increasing Subsequence Length: {lis}")
print(f"Runtime: {end_time - start_time:.5f} seconds")

test_longest_increasing_subsequence(longest_increasing_subsequence)


