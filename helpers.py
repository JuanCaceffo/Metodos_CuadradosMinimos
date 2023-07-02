from abc import ABC, abstractmethod
import sympy as sp
import pandas as pd
import numpy as np
import re

np.set_printoptions(suppress=True)

class Presicion:
    PRECISION = 2

    @classmethod
    def cambiarPresicionRepresentacion(cls, errorTolerado: float):
        cls.validarFlotante(errorTolerado)
        cls.validarFormatoError(errorTolerado)
        cls.PRECISION = cls.digitosDePresicion(errorTolerado)

    @classmethod
    def digitosDePresicion(cls, valor) -> int:
        valorTXT = cls.decimalASTR(valor)
        return len(valorTXT.split('.')[-1])

    @classmethod
    def presicionActual(cls) -> int:
        return cls.PRECISION

    @staticmethod
    def validarFlotante(numero):
        try:
            float(numero)
        except:
            raise TypeError(f"Se esperaba float y se obtuvo {type(numero).__name__}.")
    
    @classmethod
    def validarFormatoError(cls, numero):
        patron = r"^(0\.0+1|0\.1)$"        
        if not bool(re.match(patron, cls.decimalASTR(numero))):
            raise ValueError(f"Se esperaba un número del tipo 0.01 y se obtuvo {numero}.")
    
    @staticmethod
    def decimalASTR(numero):
        decimales = abs(int(np.log10(numero)))
        return '{:.{}f}'.format(numero, decimales)

#Clase general regresiones lineal, cuadratica y exponencial
class Regresion:
    df = pd.DataFrame
    x = sp.symbols('x')                                          # Simbolo X para usar en las fórmulas
    n = 0   

    def __init__(self, df:pd.DataFrame) -> None:

        if df.shape[1] != 2:
            raise ValueError(f"Se esperaba una lista con dos columnas y se obtuvo una con {len(df.shape[1])}.")

        self.df = df                                            # Dataframe a global   
        self.n = df.shape[0]                                    # Cantidad de registros para usar en promedios

    def promediar(self, col):
        return np.mean(col)

    def prodXY(self):
        return self.x_col()*self.y_col()
            
    def x2(self):
        return self.x_col()**2

    def y2(self):
        return self.y_col()**2

    def x_col(self):
        return self.df.iloc[:,0]

    def y_col(self):
        return self.df.iloc[:,1]

    def denominador(self):
        den = self.promediar(self.x2())-self.promediar(self.x_col())**2
        self.validarDenominador(den)
        return den

    def obtenerEcuacionSimbolica():
        return None

    def obtenerEcuacion(self):
        return sp.lambdify(self.x,self.obtenerEcuacionSimbolica(), modules=['numpy'])

    def pearsonError(self, rounded=False):
        coef =  self.numeradorPearson()/self.denominadorPearson()
        self.validarPearson(coef)
        return round(coef, Presicion.presicionActual()) if rounded else coef

    def denominadorPearson(self):
        den = np.sqrt(self.promediar(self.x2())-self.promediar(self.x_col())**2)*np.sqrt(self.promediar(self.y2())-self.promediar(self.y_col())**2)
        self.validarDenominador(den)
        return den

    def numeradorPearson(self):
        return self.promediar(self.prodXY())-self.promediar(self.x_col())*self.promediar(self.y_col())
    
    def validarDenominador(self, num):
        if num == 0:
            raise ValueError(f"El denominador de la función para calcular m dio cero")
    
    def validarPearson(self, num):
        if not -1 <= num <= 1:
            raise ValueError(f"El coeficiente debería estar entre -1 y 1 pero resultó en {num}")

    def imprimirEcuacion(self):
        # Define el patrón regex para encontrar números en la ecuación
        pattern = r"[-+]?\d+\.\d+"

        # Encuentra todos los números en la ecuación usando el patrón regex
        numbers = re.findall(pattern, str(self.obtenerEcuacionSimbolica()))

        # Redondea cada número a 4 decimales y conviértelo a cadena
        rounded_numbers = [str(round(float(num), Presicion.presicionActual())) for num in numbers]

        # Reemplaza los números originales por los números redondeados en la ecuación
        return sp.sympify(re.sub(pattern, lambda m: rounded_numbers.pop(0), str(self.obtenerEcuacionSimbolica())))


#Clase para calcular el m y b de la recta para usar en regresión lineal
class RegresionLinear(Regresion):
            
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__(df)

    def numeradorM(self):
        return self.promediar(self.prodXY())-self.promediar(self.x_col())*self.promediar(self.y_col())
    
    def numeradorB(self):
        return self.promediar(self.x2())*self.promediar(self.y_col())-self.promediar(self.x_col())*self.promediar(self.prodXY())
    
    def calcularM(self, rounded=False):
        return round(self.numeradorM()/self.denominador(),Presicion.presicionActual()) if rounded else self.numeradorM()/self.denominador()

    def calcularB(self, rounded=False):
        return round(self.numeradorB()/self.denominador(),Presicion.presicionActual()) if rounded else self.numeradorB()/self.denominador()
                
    def obtenerEcuacionSimbolica(self):
        equation = self.calcularM()*self.x+self.calcularB()
        return equation
    
class RegresionCuadratica(RegresionLinear):
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__(df)
    
    def x_col(self):
        return np.log(super().x_col())

    def y_col(self):
        return np.log(super().y_col())

    def calcularB(self, rounded=False):
        temp = super().calcularB()
        return round(np.e**temp,Presicion.presicionActual()) if rounded else np.e**temp
        
    def obtenerEcuacionSimbolica(self):
        equation = self.calcularB()*self.x**self.calcularM()
        return equation

class RegresionExponencial(RegresionCuadratica):
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__(df)

    def x_col(self):
        return self.df.iloc[:,0]
        
    def obtenerEcuacionSimbolica(self):
        equation = self.calcularB()*sp.E**(self.calcularM()*self.x)
        return equation

# Interfaz
class DerivadaAproximada(ABC):
    def __init__(self, funcion):
        self.funcion = funcion
                
    def execute(self, x):   
        return self._calcular(x)
                       
    @abstractmethod
    def _calcular(self, xi):
        pass

class DerivadaAdelante(DerivadaAproximada):
    def _calcular(self, x:int):
        return (self.funcion(self.xiDER(x)) - self.funcion(self.xi(x))) / abs(self.xiDER(x) - self.xi(x))

class DerivadaAtras(DerivadaAproximada):
    def _calcular(self, x:int):
        return (self.funcion(self.xi(x)) - self.funcion(self.xiIZQ(x))) / abs( self.xi(x) - self.xiIZQ(x))

class DerivadaCentral(DerivadaAproximada):
    def _calcular(self, x:int):
        return (self.funcion(self.xiDER(x)) - self.funcion(self.xiIZQ(x))) / abs(self.xiDER(x) - self.xiIZQ(x))

class DiferenciasNumericas:
    def __init__(self, puntos:pd.Series, funcion:Regresion):
        self.last = puntos.index[-1]
        self.puntos = puntos
        self.funcion = funcion

    def calcular(self, x:int):
       return self.estrategia.execute(x)

    def usarMejorEstrategia(self, x:int):
        match x:
            case 0:
                return DerivadaAdelante(self.funcion)
            case self.last:
            case _:
                action-default
        