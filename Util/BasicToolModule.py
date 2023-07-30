import sys


class BasicTool:
    def __init__(self):
        self.pTime = 0

    def countFps(self, my_time):
        cTime = my_time
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime

        return fps

    @staticmethod
    def get_base_url():
        return sys.path[1]
