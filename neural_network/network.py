import struct
from layer.layer import NeuralLayer
from algo.matrix import Matrix
from algo.sigmoid import sigmoid,mean_abs



class NeuralWork:
    def __init__(self, hidden=1, size=100) -> None:
        self.input = NeuralLayer(size)
        self.output = NeuralLayer(size)
        prev = self.input
        self.hidden = hidden
        for _ in range(hidden):
            prev.next = NeuralLayer(size)
            prev.next.prev = prev
            prev = prev.next
        prev.next = self.output
        self.output.prev = prev

    def predict(self, input: Matrix):
        t_flg = len(input) == 1
        if t_flg:
            input = input.T()
        self.input.predict(input)
        out = self.output.out
        if isinstance(out,Matrix):
            if t_flg:
                out = out.T()
        return out

    def learn(self, rate: float, learn_set: list[Matrix], correct_set: list[Matrix]):
        count = 0
        while count < 1:
            for learn,correct in zip(learn_set,correct_set):
                if len(learn) == 1:
                    learn = learn.T()   
                if len(correct) == 1:
                    correct = correct.T()
                out = self.predict(learn)
                if isinstance(out, Matrix):
                    err = out - correct
                    print('train_step', count, 'mae', mean_abs(err))
                self.output.learn(self.output.out - correct,rate)
            count += rate
            
    def __bytes__(self):
        bys = b""
        bys += int(self.hidden).to_bytes(8, byteorder='big', signed=True)
        bys += int(self.input.size).to_bytes(8, byteorder='big', signed=True)
        
        layer = self.input.next
        while layer != None and isinstance(layer,NeuralLayer):
            bys += layer.__bytes__()
            layer = layer.next
        return bys
            
    @classmethod
    def from_bytes(cls, bys):
        """从字节流创建神经网络"""
        hidden = int.from_bytes(bys[:8], byteorder='big', signed=True)
        size = int.from_bytes(bys[8:16], byteorder='big', signed=True)
        network = cls(hidden, size)
        pos = 16
        prev_l = None
        for _ in range(hidden + 2):
            layer = NeuralLayer.from_bytes(bys[pos:])
            pos += int.from_bytes(bys[pos:pos+8], byteorder='big', signed=True)
            if prev_l != None:
                prev_l.next = layer
                layer.prev = prev_l
            else:
                network.input = layer
            prev_l = layer
        network.output = prev_l
        return network
            
    def save(self, path: str):
        with open(path, 'wb') as f:
            f.write(self.__bytes__())
            
    def load(self, path: str):
        with open(path, 'rb') as f:
            bys = f.read()
            network = NeuralWork.from_bytes(bys)
        return network
