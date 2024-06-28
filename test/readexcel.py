import openpyxl as opxl

book = opxl.load_workbook("blade_parameter.xlsx", read_only=True)
sheet = book["Sheet1"]
datarange = sheet["H8":"H47"]
print(datarange[0][0].value)

book.close()