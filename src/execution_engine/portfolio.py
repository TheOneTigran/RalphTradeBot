"""
portfolio.py — Управление рисками, позициями и балансом.
"""

from typing import List

from .config import STARTING_BALANCE, RISK_PER_TRADE_PCT, TAKER_FEE_PCT, MAX_OPEN_POSITIONS
from .models import SimulatedOrder, OrderStatus


class PortfolioManager:
    def __init__(self):
        self.balance: float = STARTING_BALANCE
        self.peak_balance: float = STARTING_BALANCE
        self.max_drawdown_pct: float = 0.0
        
        self.history: List[SimulatedOrder] = []

    def can_open_position(self, active_orders_count: int) -> bool:
        if active_orders_count >= MAX_OPEN_POSITIONS:
            return False
        # Проверим, что баланс не улетел в 0
        if self.balance <= 0:
            return False
        return True

    def calculate_position_size(self, entry: float, sl: float) -> float:
        """
        Position_Size = (Balance * 0.01) / abs(Entry_Price - Stop_Loss).
        Возвращает размер позиции в контрактах (монетах).
        """
        # Мы готовы потерять вот столько долларов:
        risk_usd = self.balance * (RISK_PER_TRADE_PCT / 100.0)
        
        price_diff = abs(entry - sl)
        if price_diff == 0:
            return 0.0
            
        size_coins = risk_usd / price_diff
        return size_coins

    def apply_result(self, order: SimulatedOrder):
        """
        Пересчитывает баланс после закрытия сделки.
        """
        if order.status not in [OrderStatus.FILLED_TP, OrderStatus.FILLED_SL]:
            self.history.append(order)
            return

        size = order.position_size
        entry = order.fill_price
        exit_p = order.close_price

        # Gross PnL
        if order.is_long:
            gross_pnl = (exit_p - entry) * size
        else:
            gross_pnl = (entry - exit_p) * size

        # Taker Fee (вычитаем и при входе, и при выходе)
        entry_fee = (entry * size) * (TAKER_FEE_PCT / 100.0)
        exit_fee = (exit_p * size) * (TAKER_FEE_PCT / 100.0)
        
        net_pnl = gross_pnl - entry_fee - exit_fee
        order.pnl_usd = round(net_pnl, 2)
        
        self.balance += net_pnl
        self.history.append(order)
        
        # Обновляем Drawdown
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
            
        current_dd = (self.peak_balance - self.balance) / self.peak_balance * 100.0
        if current_dd > self.max_drawdown_pct:
            self.max_drawdown_pct = current_dd
            
        order.drawdown_pct = round(current_dd, 2)

    def get_stats(self) -> dict:
        wins = sum(1 for o in self.history if o.status == OrderStatus.FILLED_TP)
        losses = sum(1 for o in self.history if o.status == OrderStatus.FILLED_SL)
        total = wins + losses
        winrate = round(wins / total * 100, 2) if total > 0 else 0.0
        
        return {
            "Final Balance": round(self.balance, 2),
            "Max Drawdown %": round(self.max_drawdown_pct, 2),
            "Total Trades": total,
            "Win Rate %": winrate
        }
