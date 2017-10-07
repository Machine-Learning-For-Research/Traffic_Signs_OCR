import os


class Properties:
    def __init__(self, file_name):
        self.file_name = file_name
        self.dict = {}
        if os.path.exists(file_name):
            with open(file_name) as f:
                for line in f:
                    strs = line.split('=')
                    self.dict[strs[0].strip()] = strs[1].strip()

    def get(self, key):
        return self.dict[key]
