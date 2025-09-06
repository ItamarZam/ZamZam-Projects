import gradio as gr
from accounts import Account

# Create a single account for demonstration
account = None

def create_account(username: str, initial_deposit: float):
    global account
    account = Account(username, initial_deposit)
    return f"Account created for {username} with initial deposit of ${initial_deposit:.2f}"

def deposit_funds(amount: float):
    if account is None:
        return "No account exists. Please create an account first."
    account.deposit(amount)
    return f"Deposited: ${amount:.2f}. New balance: ${account.balance:.2f}"

def withdraw_funds(amount: float):
    if account is None:
        return "No account exists. Please create an account first."
    try:
        account.withdraw(amount)
        return f"Withdrew: ${amount:.2f}. New balance: ${account.balance:.2f}"
    except ValueError as e:
        return str(e)

def buy_shares(symbol: str, quantity: int):
    if account is None:
        return "No account exists. Please create an account first."
    try:
        account.buy_shares(symbol.upper(), quantity)
        return f"Bought {quantity} shares of {symbol.upper()}. New balance: ${account.balance:.2f}"
    except ValueError as e:
        return str(e)

def sell_shares(symbol: str, quantity: int):
    if account is None:
        return "No account exists. Please create an account first."
    try:
        account.sell_shares(symbol.upper(), quantity)
        return f"Sold {quantity} shares of {symbol.upper()}. New balance: ${account.balance:.2f}"
    except ValueError as e:
        return str(e)

def get_portfolio_value():
    if account is None:
        return "No account exists. Please create an account first."
    return f"Total portfolio value: ${account.calculate_portfolio_value():.2f}"

def get_profit_loss():
    if account is None:
        return "No account exists. Please create an account first."
    return f"Profit/Loss: ${account.get_profit_loss():.2f}"

def get_holdings():
    if account is None:
        return "No account exists. Please create an account first."
    return f"Current holdings: {account.get_holdings()}"

def get_transactions():
    if account is None:
        return "No account exists. Please create an account first."
    return f"Transactions: {account.get_transactions()}"

with gr.Blocks() as ui:
    gr.Markdown("### Trading Account Management System")
    username = gr.Textbox(label="Username")
    initial_deposit = gr.Number(label="Initial Deposit", value=1000)
    create_btn = gr.Button("Create Account")
    create_output = gr.Textbox(label="Create Account Result", interactive=False)
    
    create_btn.click(create_account, inputs=[username, initial_deposit], outputs=create_output)

    deposit_amount = gr.Number(label="Deposit Amount")
    deposit_btn = gr.Button("Deposit Funds")
    deposit_output = gr.Textbox(label="Deposit Result", interactive=False)
    
    deposit_btn.click(deposit_funds, inputs=deposit_amount, outputs=deposit_output)

    withdraw_amount = gr.Number(label="Withdraw Amount")
    withdraw_btn = gr.Button("Withdraw Funds")
    withdraw_output = gr.Textbox(label="Withdraw Result", interactive=False)
    
    withdraw_btn.click(withdraw_funds, inputs=withdraw_amount, outputs=withdraw_output)

    buy_symbol = gr.Textbox(label="Share Symbol (e.g., AAPL)")
    buy_quantity = gr.Number(label="Quantity to Buy")
    buy_btn = gr.Button("Buy Shares")
    buy_output = gr.Textbox(label="Buy Shares Result", interactive=False)
    
    buy_btn.click(buy_shares, inputs=[buy_symbol, buy_quantity], outputs=buy_output)

    sell_symbol = gr.Textbox(label="Share Symbol (e.g., AAPL)")
    sell_quantity = gr.Number(label="Quantity to Sell")
    sell_btn = gr.Button("Sell Shares")
    sell_output = gr.Textbox(label="Sell Shares Result", interactive=False)
    
    sell_btn.click(sell_shares, inputs=[sell_symbol, sell_quantity], outputs=sell_output)

    port_value_btn = gr.Button("Get Portfolio Value")
    port_value_output = gr.Textbox(label="Portfolio Value", interactive=False)
    
    port_value_btn.click(get_portfolio_value, outputs=port_value_output)

    profit_loss_btn = gr.Button("Get Profit/Loss")
    profit_loss_output = gr.Textbox(label="Profit/Loss", interactive=False)
    
    profit_loss_btn.click(get_profit_loss, outputs=profit_loss_output)

    holdings_btn = gr.Button("Get Current Holdings")
    holdings_output = gr.Textbox(label="Current Holdings", interactive=False)
    
    holdings_btn.click(get_holdings, outputs=holdings_output)

    transactions_btn = gr.Button("Get Transaction History")
    transactions_output = gr.Textbox(label="Transaction History", interactive=False)
    
    transactions_btn.click(get_transactions, outputs=transactions_output)

ui.launch()