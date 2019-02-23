from tkinter import *


class GUI:
    root = None

    def __init__(self):
        self.root = Tk()
        self.root.title("Aircraft Identification")
        self.root.state('zoomed')

    def home(self):
        home = Label(self.root, text="Home")
        home.pack()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    gui = GUI()
    gui.home()
    gui.run()
