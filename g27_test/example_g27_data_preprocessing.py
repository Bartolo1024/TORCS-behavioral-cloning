from random import randint
class TestInput(object):
    def __init__(self):
        with open('g27_test/g_27_random_outputs.txt', 'rb') as file:
            f = file.readlines()

        self.data = []

        for ind, row in enumerate(f):
            row = row.decode("utf-8")
            splt = row.split()
            parsed_row = [splt[0], splt[1], ' '.join(splt[2: len(splt)])]
            self.data.append(parsed_row)

    def get_random_signal(self):
        return self.data[randint(0, self.data.__len__() - 1)] # ret hex, val, button