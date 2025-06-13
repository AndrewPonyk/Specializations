def calculate_average(numbers):
    return sum(numbers) / len(numbers)

def test_calculate_average():
    assert calculate_average([1, 2, 3]) == 2
    assert calculate_average([1, 2, 3, 4]) == 2.5
    assert calculate_average([10, 20, 30]) == 20
    assert calculate_average([10, 20, 30, 40]) == 25

def calculate_median(numbers):
    numbers.sort()
    if len(numbers) % 2 == 0:
        return (numbers[len(numbers) // 2] + numbers[len(numbers) // 2 - 1]) / 2
    else:
        return numbers[len(numbers) // 2]

# A function that returns the mode of a list of numbers
def calculate_mode(numbers):
    if not numbers:
        return []
    counter = {}
    for number in numbers:
        if number in counter:
            counter[number] += 1
        else:
            counter[number] = 1
    max_occurrences = max(counter.values())
    mode = [number for number, occurrences in counter.items() if occurrences == max_occurrences]
    return mode

def bubble_sort(numbers):
    for i in range(len(numbers) - 1):
        for j in range(len(numbers) - 1 - i):
            if numbers[j] > numbers[j + 1]:
                numbers[j], numbers[j + 1] = numbers[j + 1], numbers[j]
    return numbers

def calculate_std_dev(numbers):
    average = calculate_average(numbers)
    variance = sum((number - average) ** 2 for number in numbers) / len(numbers)
    return variance ** 0.5