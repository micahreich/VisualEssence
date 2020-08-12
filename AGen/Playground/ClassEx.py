class Clock:
    def __init__(self):
        self.seconds = 0
        self.minutes = 0
        self.hours = 0

    def run_clock(self):
        while self.seconds < 3600:
            self.seconds += 1
            if self.seconds % 60 == 0:
                self.minutes += 1

            if self.seconds % 3600 == 0:
                self.hours += 1
                print(self.hours)


Clock().run_clock()