# -*- coding: utf-8 -*-
"""Convert the clinical CRF spreadsheet (.xls/.xlsx) to crf.csv used by analysis10-14.

Place the CRF workbook in data/ and run:
    python scripts/convert_crf.py
The derived crf.csv is written to data/ (restricted clinical data; never commit it).
"""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = os.path.join(ROOT, 'data', '临床CRF表.xls')
DST = os.path.join(ROOT, 'data', 'crf.csv')

df = pd.read_excel(SRC)
print('shape:', df.shape)
print('columns:', list(df.columns))
df.to_csv(DST, index=False, encoding='utf-8-sig')
print('saved', DST)
