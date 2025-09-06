# accounts.py

class Account:
    def __init__(self, username: str, initial_deposit: float):
        """
        Initializes a new trading account for the user.
        
        :param username: The username of the account holder.
        :param initial_deposit: The opening balance for the account.
        """
        self.username = username
        self.balance = initial_deposit
        self.holdings = {}  # Dictionary to hold the shares owned {symbol: quantity}
        self.transactions = []  # List to hold transaction history
        self.initial_deposit = initial_deposit

    def deposit(self, amount: float):
        """
        Deposits funds into the user's account.
        
        :param amount: The amount to deposit.
        """
        if amount <= 0:
            raise ValueError("Deposit amount must be positive.")
        self.balance += amount
        self.transactions.append(f"Deposited: {amount}")

    def withdraw(self, amount: float):
        """
        Withdraws funds from the user's account.
        
        :param amount: The amount to withdraw.
        """
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive.")
        if amount > self.balance:
            raise ValueError("Insufficient funds for withdrawal.")
        self.balance -= amount
        self.transactions.append(f"Withdrew: {amount}")

    def buy_shares(self, symbol: str, quantity: int):
        """
        Records the purchase of shares for the specified symbol.
        
        :param symbol: The stock symbol to purchase.
        :param quantity: The number of shares to buy.
        """
        if quantity <= 0:
            raise ValueError("Quantity must be positive.")
        share_price = get_share_price(symbol)
        total_cost = share_price * quantity
        if total_cost > self.balance:
            raise ValueError("Insufficient funds to buy shares.")
        
        # Update balance and holdings
        self.balance -= total_cost
        if symbol in self.holdings:
            self.holdings[symbol] += quantity
        else:
            self.holdings[symbol] = quantity
        self.transactions.append(f"Bought {quantity} shares of {symbol}")

    def sell_shares(self, symbol: str, quantity: int):
        """
        Records the sale of shares for the specified symbol.
        
        :param symbol: The stock symbol to sell.
        :param quantity: The number of shares to sell.
        """
        if quantity <= 0:
            raise ValueError("Quantity must be positive.")
        if symbol not in self.holdings or self.holdings[symbol] < quantity:
            raise ValueError("Insufficient shares to sell.")
        
        share_price = get_share_price(symbol)
        total_revenue = share_price * quantity
        self.balance += total_revenue
        self.holdings[symbol] -= quantity
        if self.holdings[symbol] == 0:
            del self.holdings[symbol]
        self.transactions.append(f"Sold {quantity} shares of {symbol}")

    def calculate_portfolio_value(self) -> float:
        """
        Calculates the total value of the user's portfolio based on current share prices.
        
        :return: Total portfolio value.
        """
        total_value = self.balance
        for symbol, quantity in self.holdings.items():
            total_value += get_share_price(symbol) * quantity
        return total_value

    def calculate_profit_loss(self) -> float:
        """
        Calculates profit or loss from the initial deposit.
        
        :return: Profit or loss amount.
        """
        return self.calculate_portfolio_value() - self.initial_deposit

    def get_holdings(self) -> dict:
        """
        Returns the user's current holdings.
        
        :return: A dictionary of holdings {symbol: quantity}.
        """
        return self.holdings

    def get_profit_loss(self) -> float:
        """
        Returns the user's profit or loss.
        
        :return: Profit or loss amount.
        """
        return self.calculate_profit_loss()

    def get_transactions(self) -> list:
        """
        Returns the list of transactions made by the user.
        
        :return: A list of transaction history.
        """
        return self.transactions

def get_share_price(symbol: str) -> float:
    """
    Simulated function to get the current share price.
    
    :param symbol: The stock symbol for which price is needed.
    :return: Current share price.
    """
    prices = {
        'AAPL': 150.00,
        'TSLA': 700.00,
        'GOOGL': 2800.00,
    }
    return prices.get(symbol, 0.0)