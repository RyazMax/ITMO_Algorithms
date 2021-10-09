import numpy as np

CONST = 5


def const(_: np.ndarray) -> int:
    return CONST


def sum(x: np.ndarray):
    return np.sum(x)


def mul(x: np.ndarray):
    return np.prod(x)


def polynom(x: np.ndarray):
    return np.sum(x * (1.5 ** np.arange(len(x), dtype=np.float128)))


def polynom_horner(v: np.ndarray):
    x = np.float128(1.5)
    s = np.float128(0)
    for vk in v[::-1]:
        s = s * x + vk
    return s


def bubble_sort(x: np.ndarray):
    res = list(x)
    for i in range(len(res)):
        for j in range(len(res) - i - 1):
            if res[j] > res[j + 1]:
                res[j], res[j + 1] = res[j + 1], res[j]
    return np.array(res)


def _partition(array, start, end):
    pivot = array[start]
    low = start + 1
    high = end

    while True:
        # If the current value we're looking at is larger than the pivot
        # it's in the right place (right side of pivot) and we can move left,
        # to the next element.
        # We also need to make sure we haven't surpassed the low pointer, since that
        # indicates we have already moved all the elements to their correct side of the pivot
        while low <= high and array[high] >= pivot:
            high = high - 1

        # Opposite process of the one above
        while low <= high and array[low] <= pivot:
            low = low + 1

        # We either found a value for both high and low that is out of order
        # or low is higher than high, in which case we exit the loop
        if low <= high:
            array[low], array[high] = array[high], array[low]
            # The loop continues
        else:
            # We exit out of the loop
            break

    array[start], array[high] = array[high], array[start]

    return high


def _quick_sort(array, start, end):
    if start >= end:
        return

    p = _partition(array, start, end)
    _quick_sort(array, start, p-1)
    _quick_sort(array, p+1, end)


def quick_sort(x: np.ndarray):
    res = list(x)
    _quick_sort(res, 0, len(res) - 1)
    return np.array(res)


def tim_sort(x: np.ndarray):
    return np.sort(x, kind='mergesort')  # use timsort for not integer dtype


def matrix_product(matrixs):
    a, b = matrixs
    b = b.T
    return np.array([[a[i].dot(b[j]) for j in range(len(b))] for i in range(len(a))])
