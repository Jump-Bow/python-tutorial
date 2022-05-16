from win32com import client
  
# Open Microsoft Excel
excel = client.Dispatch("Excel.Application")
  
# Read Excel File
sheets = excel.Workbooks.Open(r'C:/Users/PDFTEST/file.xlsx')
# work_sheets = sheets.Worksheets[0]
  
# Convert into PDF File
sheets.ExportAsFixedFormat(0, r'C:/Users/PDFTEST/file.pdf')
sheets.close()