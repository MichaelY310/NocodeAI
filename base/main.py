class Father:
    def __init__(self, a, b, c=0, d="d"):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def p(self):
        print(self.a, self.b, self.c, self.d)


f = Father(114, 514)

class Son(Father):
    def __init__(self, *args):
        super(Son, self).__init__(*args)
        name = "son"

s = Son(1919, 810)

f.p()

s.p()



