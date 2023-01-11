from flask import Flask, render_template, json, request
from WebView.ObjectsApp.DAL.VideoDAL import VideoDAL
from config import Config, MySQL_DB

app = Flask(__name__)
app.config.from_object(Config)
MySQL_DB.mysql.init_app(app)


@app.route("/",  methods = ['GET'])
def main():
   
    video_id = request.args.get('video_id')

    if video_id == 'All':
        video_id = None

    context = {}
    context['videos'] = []
    context['videos'].append({'video_id': 'All', 'video_name': 'All'})
    context['results'] = []
    context['error_msg'] = ""
    try:
        videos = VideoDAL().getAllVideos()
        
        for item in videos:
            if video_id != None and item['video_id'] == int(video_id):
                item['selected'] = True
            else:
                item['selected'] = False
            context['videos'].append(item)

       
        context['results'] = VideoDAL().getAllInferences(video_id)
   

    
    except Exception as e:
        print('----------ERROR------------')
        print(str(e))
        context['error_msg'] = str(e)

    return render_template('index.html', context = context)






if __name__ == "__main__":

    app.run(debug=True)
    # VideoDAL().getAllVideos()