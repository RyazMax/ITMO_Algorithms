import argparse
import timeit

import func
import generator

DEFAULT_EXPERIMENTS = 5
DEFAULT_START = 1
DEFAULT_END = 2000

tasks = [
    (generator.vector, func.const),  # 0
    (generator.vector, func.sum),  # 1
    (generator.vector, func.mul),  # 2
    (generator.vector, func.polynom),  # 3
    (generator.vector, func.polynom_horner),  # 4
    (generator.vector, func.bubble_sort),  # 5
    (generator.vector, func.quick_sort),  # 6
    (generator.vector, func.tim_sort),  # 7
    (generator.matrixes, func.matrix_product),  # 8
]

def time_iter(iter, task, experiments):
    generator, func = task
    data = generator(iter)
    return timeit.timeit(lambda: func(data), number=experiments) / experiments


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', help='minimum size of data', type=int, default=DEFAULT_START)
    parser.add_argument('--end', help='maximum size of data', type=int, default=DEFAULT_END)
    parser.add_argument('--task', help='Number of task to time', type=int, required=True, choices=list(range(len(tasks))))
    parser.add_argument('--output', help='Output file to write data to', type=str, required=True)
    parser.add_argument('--experiments', help='Number of experiments to perform', type=int, default=DEFAULT_EXPERIMENTS)

    args = parser.parse_args()

    task = tasks[args.task]
    with open(args.output, 'w') as fout:
        for i in range(args.start, args.end):
            print(f'Iter: {i} / {args.end}')
            time = time_iter(i, task, args.experiments)
            fout.write(f'{i}, {time}\n')