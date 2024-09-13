import pandas as pd
import numpy as np

def transaction_classification(group):
    '''
    Function to classify a group of transactions and return a summary based on the number and type of transactions.
    
    Args:
    group (pd.DataFrame): A DataFrame containing transactions for a single user or entity.

    Returns:
    pd.Series: A Series containing the classification category, count of subsequent transactions, and the total amount of those transactions.
    '''
    if len(group) < 2:
        return pd.Series({'category': 'Normal', 'count': 0, 'amount': 0, 'merchant': None})
    
    first_transaction = group.iloc[0]
    subsequent_transactions = group.iloc[1:]
    
    if (subsequent_transactions['transactionType'] == 'REVERSAL').any():
        category = 'Reversal'
    elif (subsequent_transactions['transactionType'] == 'PURCHASE').all():
        category = 'Multi-swipe'
    else:
        category = 'Normal'
    
    return pd.Series({
        'category': category,
        'count': len(subsequent_transactions),
        'amount': subsequent_transactions['transactionAmount'].sum(),
        'merchant': first_transaction['merchantName']
    })
