def run(func, **kwargs):
    print("hello world")
    if func.__name__ == "test":
        func(**kwargs)
    print("1")

def test(a, b):
    print(a + b)
    
run(test, a=3, b=5)
