class Progress:
    def __init__(self, size:tuple, project:str):
        self.size = size
        self.project = project
        self.flagged = 0
        self.unique_flagged = 0

    def get_size(self):
        return self.size
    def get_project(self):
        return self.project

    def set_flagged(self, flagged):
        self.flagged = int(flagged)
    
    def get_flagged(self):
        return self.flagged
    
    def set_unique_flagged(self, unique_flagged):
        self.unique_flagged = unique_flagged
    
    def get_unique_flagged(self):
        return self.unique_flagged