from tkinter import N
import random as r
import struct

class Matrix:
    def __init__(self, data):
        self.data = [[float(x) for x in row] for row in  data]

    def sum(self):
        """矩阵按行求和"""
        m = [self.__len__]
        for i in range(len(self)):
            for j in range(len(self[0])):
                m[i] += self[i][j]
        return m

    @classmethod
    def from_int(cls, n, down=0.0, up=0.0):
        """
        从整数创建 nxn 矩阵

        参数:
            n (int): 方阵的阶数
            down (float): 随机数下限，默认为 0.0
            up (float): 随机数上限，默认为 0.0
        """
        return cls([[r.uniform(down, up) for i in range(n)] for j in range(n)])

    @classmethod
    def from_coord(cls, x, y, down=0.0, up=0.0):
        """
        从整数创建 x×y 矩阵

        参数:
            x (int): 矩阵的行数
            y (int): 矩阵的列数
            down (float): 随机数下限，默认为 0.0
            up (float): 随机数上限，默认为 0.0
        """
        return cls([[r.uniform(down, up) for i in range(y)] for j in range(x)])

    def T(self):
        """矩阵转置"""
        rows = len(self.data)
        cols = len(self.data[0]) if rows > 0 else 0
        # 创建新矩阵存放转置结果，避免原地修改
        transposed = [[self.data[j][i] for j in range(rows)] for i in range(cols)]
        return Matrix(transposed)

    def __copy__(self):
        return [[n for n in roll] for roll in self.data]

    # 获取索引值：obj[index]
    def __getitem__(self, index):
        return self.data[index]

    # 设置索引值：obj[index] = value
    def __setitem__(self, index, value):
        self.data[index] = value

    # 删除索引值：del obj[index]
    def __delitem__(self, index):
        del self.data[index]

    def __str__(self):
        if not self.data or not self.data[0]:
            return "[]"

        # 找到每列的最大宽度
        col_widths = [0] * len(self.data[0])
        for row in self.data:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(f"{cell:.4f}"))

        # 格式化矩阵字符串
        s = "[\n"
        for row in self.data:
            s += "  ["
            for i, cell in enumerate(row):
                s += f"{cell:<{col_widths[i]}.4f} "
            s = s.rstrip()
            s += "]\n"
        s += "]"
        return s

    def __bytes__(self):
        bys = b""
        bys += len(self).to_bytes(8, byteorder='big')
        bys += len(self[0]).to_bytes(8, byteorder='big')
        for row in self.data:
            for cell in row:
                bys += struct.pack('d', cell)
        return bys
    
    @classmethod
    def from_bytes(cls, bys):
        """从字节流创建矩阵"""
        nrows = int.from_bytes(bys[:8], byteorder='big')
        ncols = int.from_bytes(bys[8:16], byteorder='big')
        data = []
        for i in range(nrows):
            row = []
            for j in range(ncols):
                offset = 16 + i * ncols * 8 + j * 8
                cell = struct.unpack('d', bys[offset:offset+8])[0]
                row.append(cell)
            data.append(row)
        return cls(data)
    
    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        """__add__ 加法 计算 a + b
        """        
        
        # 标量 + 矩阵 | 矩阵 + 标量
        if isinstance(other, (int, float)):
            return Matrix([[n + other for n in roll] for roll in self.data])
        
        # 矩阵 + 矩阵 逐元素相加, 要求矩阵维度相同
        if isinstance(other, Matrix):
            if len(self) != len(other) or len(self[0]) != len(other[0]):
                raise ValueError("矩阵维度不匹配，无法逐元素相加")
            return Matrix([[a + b for a, b in zip(self_row, other_row)] for self_row, other_row in zip(self.data, other.data)])
        return NotImplemented
        
    def __radd__(self, other):
        """__radd__ 右加法 同左加法，适配标量计算
        """  
        return self.__add__(other)
    

    def __sub__(self, other):
        """__sub__ 减法 计算 a - b
        """        
        
        # 标量 - 矩阵
        if isinstance(other, (int, float)):
            return Matrix([[n - other for n in roll] for roll in self.data])
        
        # 矩阵 - 矩阵 逐元素相减, 要求矩阵维度相同
        if isinstance(other, Matrix):
            if len(self) != len(other) or len(self[0]) != len(other[0]):
                raise ValueError("矩阵维度不匹配，无法逐元素相减")
            return Matrix([[a - b for a, b in zip(self_row, other_row)] for self_row, other_row in zip(self.data, other.data)])
        return NotImplemented
        
    def __rsub__(self, other):
        """__rsub__ 右减法 计算 a - b -> 
            self是b, other是a，等价于b被a减
        """        
        
        # 标量 - 矩阵
        if isinstance(other, (int, float)):
            return Matrix([[other - n for n in roll] for roll in self.data])
        return NotImplemented

    def __mul__(self, other):
        """__mul__ 乘法 计算 a * b
            当a b都是矩阵时，进行哈德马乘积，即逐元素相乘
        """        
        
        # 标量 * 矩阵 | 矩阵 * 标量
        if isinstance(other, (int, float)):
            return Matrix([[n * other for n in roll] for roll in self.data])
        
        # 矩阵 * 矩阵 哈德马乘积, 要求矩阵维度相同
        if isinstance(other, Matrix):
            if len(self) != len(other) or len(self[0]) != len(other[0]):
                raise ValueError("矩阵维度不匹配，无法进行哈德马乘积")
            return Matrix([[a * b for a, b in zip(self_row, other_row)] for self_row, other_row in zip(self.data, other.data)])
        return NotImplemented
        
    def __rmul__(self, other):
        """__rmul__ 右乘法 同左乘法，适配标量计算
        """        
        
        return self.__mul__(other)
    
    def __matmul__(self, other):
        """__matmul__ 矩阵乘法 计算 a @ b
        """

        # 矩阵 @ 矩阵 标准矩阵乘法, 要求左矩阵列数等于右矩阵行数
        if isinstance(other, Matrix):
            if len(self[0]) != len(other):
                raise NotImplementedError("矩阵维度不匹配，无法进行矩阵乘法")
            n = len(self)
            m = len(other[0])
            s = len(self[0])
            tmp_data = [[0.0] * m for _ in range(n)]
            for i in range(n):
                row_i = self[i]
                for j in range(m):
                    acc = 0.0
                    for k in range(s):
                        acc += row_i[k] * other[k][j]
                    tmp_data[i][j] = acc

            return Matrix(tmp_data)
        return NotImplemented
        
    def __pow__(self, other):
        """__pow__ 幂运算 计算 a ** b
            当a b都是矩阵时，进行逐元素幂运算
        """        
        
        # 矩阵 ** 标量 对矩阵的每个元素进行幂运算
        if isinstance(other, (int, float)):
            return Matrix([[n ** other for n in roll] for roll in self.data])
        
        # 矩阵 ** 矩阵 要求矩阵维度相同
        if isinstance(other, Matrix):
            if len(self) != len(other) or len(self[0]) != len(other[0]):
                raise ValueError("矩阵维度不匹配，无法进行矩阵幂运算")
            return Matrix([[a ** b for a, b in zip(self_row, other_row)] for self_row, other_row in zip(self.data, other.data)])
        return NotImplemented
        
    def __rpow__(self, other):
        """__rpow__ 右幂运算 计算 a ** b -> 
            self是b, other是a，等价于b被a幂运算
        """        
        
        # 标量 ** 矩阵 对矩阵的每个元素进行幂运算
        if isinstance(other, (int, float)):
            return Matrix([[other ** n for n in roll] for roll in self.data])
        return NotImplemented

    def __truediv__(self, other):
        """__truediv__ 除法 计算 a / b
            当a b都是矩阵时，进行逐元素相除
        """        
        
        # 标量 / 矩阵 对矩阵的每个元素进行相除
        if isinstance(other, (int, float)):
            return Matrix([[n / other for n in roll] for roll in self.data])
        
        # 矩阵 / 矩阵 要求矩阵维度相同
        if isinstance(other, Matrix):
            if len(self) != len(other) or len(self[0]) != len(other[0]):
                raise ValueError("矩阵维度不匹配，无法进行矩阵相除")
            return Matrix([[a / b for a, b in zip(self_row, other_row)] for self_row, other_row in zip(self.data, other.data)])
        return NotImplemented

    def __rtruediv__(self, other):
        """__rtruediv__ 右除法 计算 a / b -> 
            self是b, other是a，等价于b被a相除
        """        
        
        # 标量 / 矩阵 对矩阵的每个元素进行相除
        if isinstance(other, (int, float)):
            return Matrix([[other / n for n in roll] for roll in self.data])
        return NotImplemented

    def __isub__(self, other):
        """__isub__ 减法赋值 计算 a -= b
        """        
        
        # 标量 -= 矩阵 对矩阵的每个元素进行相减
        if isinstance(other, (int, float)):
            for i in range(len(self.data)):
                for j in range(len(self.data[i])):
                    self.data[i][j] -= other
            return self
        
        # 矩阵 -= 矩阵 要求矩阵维度相同
        if isinstance(other, Matrix):
            if len(self) != len(other) or len(self[0]) != len(other[0]):
                raise ValueError("矩阵维度不匹配，无法进行矩阵相减")
            for i in range(len(self.data)):
                for j in range(len(self.data[i])):
                    self.data[i][j] -= other.data[i][j]
            return self
        return NotImplemented

    def __neg__(self):
        """__neg__ 负号 计算 -a
        """        
        
        return Matrix([[-n for n in roll] for roll in self.data])
