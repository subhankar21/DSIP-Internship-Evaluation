import pandas as pd
import requests
import zipfile
import io


def url_dframe(url):
    ''' 
    Extract the data (.txt file is in JSON lines format) from URL return pandas dataframe
    
    Args:
        url (str): The URL directing to the ZIP file containing .txt files.
        
    Returns:
        pd.DataFrame or None: Returns a DataFrame if the .txt file is in JSON lines format, 
        or None if no .txt files are found or the format is not JSON lines.
    '''
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        file_list = z.namelist()
        text_files = [f for f in file_list if f.endswith('.txt')]
        if text_files:
            with z.open(text_files[0]) as file:
                df = pd.read_json(file, lines=True)
        else:
            df = None
        return df
