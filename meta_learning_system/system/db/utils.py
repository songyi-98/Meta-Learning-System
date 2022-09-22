"""Utilities for databases."""
import io
import sqlite3

import numpy as np

def arr_to_text(arr):
    '''Convert np.array to TEXT in SQLite3.'''
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def text_to_arr(text):
    '''Convert TEXT to np.array in SQLite3.'''
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out, allow_pickle=True)
