from dataclasses import dataclass

class Currency:
    def to_usd(self, val:float):
        return val * self.usd_rate

    def to_gbp(self, val:float):
        return val * self.gpb_rate

    def to_gbx(self, val:float):
        return val * self.gpx_rate

    def to_brl(self, val:float):
        return val * self.brl_rate

@dataclass
class CurrencyUSD(Currency):
    brl_rate: float
    gbx_rate: float
    gbp_rate: float 
    usd_rate: float = 1

@dataclass
class CurrencyGBP(Currency):
    usd_rate: float
    brl_rate: float
    gbx_rate: float = 100
    gbp_rate: float = 1
    
@dataclass
class CurrencyGBX(Currency):
    usd_rate: float
    brl_rate: float
    gbx_rate: float = 1
    gbp_rate: float = 1/100

