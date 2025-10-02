import tqdm


class FileLoader:
    def __init__(self, file_path: str):
        self._file_path = file_path

    def create_vocab(self):
        string_list = []
        with open(self._file_path) as f:

            for line in tqdm(f, total=sum(1 for _ in open(self._file_path)), desc="Loading file"):
                print(line)
                for char in line:
                    string_list.append(char)

        idx2char = set(string_list.sort())
        char2idx = {}

        for i, char in enumerate(idx2char):
            char2idx[char] = i

        return idx2char, char2idx
