
import cx_Oracle
conn = cx_Oracle.connect("user", "password", "IP/TOPTEST")
cursor = conn.cursor()
cursor.execute("select gen01,gen02 from table where gen01 = :id",id='idname')

for gen01, gen02 in cursor:
    print("Values:", gen01, gen02)

