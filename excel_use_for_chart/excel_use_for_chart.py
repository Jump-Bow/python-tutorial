import xlwings as xw
import pythoncom


if __name__ == '__main__':  
    pythoncom.CoInitialize()
    app = xw.App(visible=False)  # 關掉顯示 EXCLE 動作
    wb = xw.Book(r'template.xlsx')  # 讀入檔案
    sht = wb.sheets['test'].charts[0]
    wb.sheets['test'].charts[0].api[1].Axes(1).MaximumScale #修改 X軸
    wb.sheets['test'].charts[0].api[1].Axes(2).MaximumScale #修改 Y軸
    wb.charts[0].api[1].ChartTitle.Text     #修改  chart title
    print (sht)


    wb.close()  # 關掉檔案
    app.quit()  # 關掉程序