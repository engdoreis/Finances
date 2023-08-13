# Wallet 
This is python package analyze transactions from the brokers below and generate report with the performance compared with market indexes.
 - Trading 212;
 - TD Ameritrade;
 - Clear

## How to use
The notebook `Wallet.ipynb` can be opened either with Visual code or Jupyter notebook.
Download the transactions statements from the broker in csv format.
Create a input config
```python
config = Input(broker=Broker.TDAMERITRADE,\
     statement_dir=f"{path_to_transactions}/transactions_td_ameritrade")
```
Execute the notebook.