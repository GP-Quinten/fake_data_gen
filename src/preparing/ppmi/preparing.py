def main_ppmi_preparing(df_ppmi, columns, drop_na=False):
    """Prepare ppmi dataframe 

    Args:
        df_ppmi (DataFrame): PPMI dataframe
        columns (list): columns to keep
        drop_na (bool, optional): Drop rows with null. Defaults to False.

    Returns:
        _type_: _description_
    """
    df_ppmi = df_ppmi[columns]
    #baseline events
    df_ppmi = df_ppmi[df_ppmi['EVENT_ID']=='BL']
    df_ppmi = df_ppmi.drop('EVENT_ID', axis=1)
    
    #APPRDX == 1
    df_ppmi = df_ppmi[df_ppmi['APPRDX']==1]
    df_ppmi = df_ppmi.drop('APPRDX', axis=1)
    
    if drop_na: 
        df_ppmi = df_ppmi.dropna(axis=0)
    df_ppmi = df_ppmi.set_index('PATNO')
    return df_ppmi