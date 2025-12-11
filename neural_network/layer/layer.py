from algo.matrix import Matrix
from algo.sigmoid import sigmoid as sgm


class NeuralLayer:
    def __init__(self, size: int = 100,prev = None,next =None) -> None:
        self.size = size
        self.weight = Matrix.from_int(size, up=1)
        self.next = next
        self.out = None
        self.inp = None
        self.prev = prev

    def predict(self, input: Matrix):
        self.inp = input
        self.out = sgm(self.weight @ input)
        if self.next != None and isinstance(self.next,NeuralLayer):
            self.next.predict(self.out)

    def learn(self, input: Matrix, rate:float):
        if self.out == None or not isinstance(self.out,Matrix) or self.inp == None:
            return
        delta = input * self.out * (1 - self.out)
        delta_prev = self.weight.T() @ delta
        grad = delta @ self.inp.T()
        self.weight -= rate * grad
        if self.prev != None and isinstance(self.prev,NeuralLayer):
            self.prev.learn(delta_prev,rate)

    def __bytes__(self):
        bys = b""
        bys += self.size.to_bytes(8, byteorder='big')
        bys += self.weight.__bytes__()
        bys = (len(bys) + 8).to_bytes(8, byteorder='big') + bys
        return bys
    
    @classmethod
    def from_bytes(cls, bys):
        """从字节流创建神经网络层"""
        size = int.from_bytes(bys[8:16], byteorder='big')
        weight = Matrix.from_bytes(bys[16:])
        layer = cls(size)
        layer.weight = weight
        return layer
