from flaskext.mysql import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from pymysql.cursors import DictCursor
from base64 import b64encode, b64decode
from config import MySQL_DB


class VideoDAL:

    def convertImagetobase64(self, image):
        if image != None:
            return b64encode(image).decode("utf-8")
        return image



    def getAllInferences(self,  video_id=None ):
        _connection = MySQL_DB().mysql.connect()
        cursor = _connection.cursor()
        results = []
        error_msg = ""
        try:
            if video_id == None:
                cursor.execute("SELECT * from video_inference_data")          
            else:
                cursor.execute("SELECT * from video_inference_data where video_id={0}".format(video_id))

            rows = cursor.fetchall()
  
            for item in rows:
                item['snapshot']  = self.convertImagetobase64(item['snapshot'])
                results.append(item)
                
        except Exception as e:
            print(str(e))         
            error_msg = str(e)
        finally:
            cursor.close()
            _connection.close()
            return results


    def getAllVideos(self):
        
        _connection = MySQL_DB().mysql.connect()
        _cursor = _connection.cursor()
        results = []
        error_msg = ""
        try:
            _cursor.execute("SELECT * from video")
            results  = _cursor.fetchall()
                    
        except Exception as e:
            print(str(e))      

        finally:
            _cursor.close()
            _connection.close()
            return results

    def insertVideo(self,id, name):
        _connection = MySQL_DB().mysql.connect()
        _cursor = _connection.cursor()
        _inserted = False
        error_msg = ""
        try:
            _query = "INSERT INTO video (video_id,video_name) VALUES (%s,%s);"
            _cursor.execute(_query, (id,name))
            _connection.commit()
            _inserted = True

        except Exception as e:
            print(str(e))
            _inserted = False

        finally:
            _cursor.close()
            _connection.close()
            return _inserted

    def insertInference(self, video):
        _connection = MySQL_DB().mysql.connect()
        _cursor = _connection.cursor()
        _inserted = False
        error_msg = ""
        try:
            _query = "INSERT INTO video_inference_data (video_id, category, confidence, upper_color, lower_color,ship_name,snapshot, start_time, span) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);"
            _cursor.execute(_query, (video['video_id'], video['category'], video['confidence'], video['color1'], video['color2'], video['ship_name'], video['snapshot'], video['start_time'], video['span']))
            _connection.commit()
            _inserted = True

        except Exception as e:
            print(str(e))
            _inserted = False

        finally: 
            _cursor.close()
            _connection.close()
            return  _inserted

