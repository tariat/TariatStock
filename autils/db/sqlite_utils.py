import sqlite3

class SqLite:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def __exit__(self):
        self.conn.close()

    # def create_table(self):
    #     cursor = self.conn.cursor()
    #     cursor.execute('''
    #     CREATE TABLE IF NOT EXISTS sitemap_data (
    #         sitemap_url TEXT,
    #         url TEXT,
    #         title TEXT,
    #         description TEXT,
    #         UNIQUE(sitemap_url, url)
    #     )
    #     ''')

    def execute(self, sql, params=None, many=False):
        try:
            if many and params:
                self.cursor.executemany(sql, params)
            elif params:
                self.cursor.execute(sql, params)
            else:
                self.cursor.execute(sql)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
            print(sql)
            self.conn.rollback()

    def get(self, sql, params=None):
        """
        table 읽기
        Args:
            sql: 실행할 sql
        Returns:
            df: dataframe
        """
        import pandas as pd
        if params:
            df = pd.read_sql(con=self.conn, sql=sql, params=params)
        else:
            df = pd.read_sql(con=self.conn, sql=sql)

        return df
        
    def get_lst(self, query, params=None):
        if params:
            self.cursor.execute(query, params)
        else:
            self.cursor.execute(query)
            
        return self.cursor.fetchall()
           
    def read_sql_to_csv(self, query, params=None):
        """
        ga_v2.py에서 사용
        """
        if params:
            self.cursor.execute(query, params)
        else:
            self.cursor.execute(query)
        
        columns = [description[0] for description in self.cursor.description]
        results = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
        
        return results
    