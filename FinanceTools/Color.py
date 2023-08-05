class Color:
    def color_negative_red(self, val):
        return "color: %s" % ("red" if val < 0 else "green")
