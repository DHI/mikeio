class FileDoesNotExist(Exception):
    def __init__(self, filename=""):
        self.message = f"File {filename} does not exist."
        super().__init__(self.message)