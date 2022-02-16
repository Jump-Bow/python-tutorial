import mariadb

def connset():
    conn = mariadb.connect(
        user="USER",
        password="密碼",
        host="IP",
        port=3306,
        # charset="utf8",
        database="testdb"
    )
    DATABASE_CONNECTION_POOLING = True
    conn.autocommit = False
    return conn
    
    
if __name__ == '__main__': 
    conn = connset()
    cursor = conn.cursor()
    sql_update_query = "select * from TABLE"
        
    cursor.execute(sql_update_query)
    result = cursor.fetchone()
    print(result[-1])
    print(type(result[-1]))