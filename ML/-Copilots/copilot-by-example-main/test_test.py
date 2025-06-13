from test import calculate_mode

def test_calculate_mode():
    # Test case 1: Single mode
    numbers = [1, 2, 2, 3, 3, 3]
    assert calculate_mode(numbers) == [3]

    # Test case 2: Multiple modes
    numbers = [1, 2, 2, 3, 3, 3, 4, 4]
    assert calculate_mode(numbers) == [3]

    # Test case 3: No mode
    numbers = [1, 2, 3, 4, 5]
    assert calculate_mode(numbers) == [1, 2, 3, 4, 5]

    # Test case 4: Empty list
    numbers = []
    assert calculate_mode(numbers) == []

    # Test case 5: Negative numbers
    numbers = [-1, -2, -2, -3, -3, -3]
    assert calculate_mode(numbers) == [-3]

    print("All test cases pass")

test_calculate_mode()