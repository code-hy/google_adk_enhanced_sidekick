
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

# Example usage:
num = 5
print(f"The factorial of {num} is {factorial(num)}")
