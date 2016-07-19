import tensorflow as tf
from flask import Flask, render_template, request
import os
#import demo
#import demo.main
host = '0.0.0.0'
port = 8999

global MOVIE_NAME
MOVIE_NAME = 'Movie Name'
ROOT_DIR = os.getcwd()
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates/resource')
app = Flask(__name__, static_folder=ASSETS_DIR)
print(ASSETS_DIR)


#@app.route('/hello/<user>')
#def hello_name(user):
#    return render_template('hello.html', name=user)


@app.route('/')
def hello_name():
    return render_template('index.html')


@app.route("/result", methods=['POST', 'GET'])
def result():
    global MOVIE_NAME
    if request.method == 'POST':
        result = request.form
        MOVIE_NAME = result['movie_name']
        movie_path = os.path.join('video_clips', result['movie_name'])
        try:
            video_clips = [clip for clip in os.listdir(os.path.join(ASSETS_DIR, movie_path))]
            clip_path = os.path.join('video_clips', result['movie_name'], video_clips[0])
            return render_template('load_movie.html', clip_path=os.path.join('resource', clip_path), MOVIE_NAME=MOVIE_NAME)
        except:
            return render_template('load_movie.html', MOVIE_NAME=MOVIE_NAME, clip_path='')
    return render_template('load_movie.html')


@app.route("/question", methods=['POST', 'GET'])
def question():
    global MOVIE_NAME
    result = request.form
    query = result['question_text']
    test_clips = {
        'first': "tt0074285.sf-119927.ef-120198.video.mp4",
        'second': "tt0074285.sf-119927.ef-120198.video.mp4",
        'third': "tt0074285.sf-119927.ef-120198.video.mp4"
    }
    answer = 'answer!!!!!'
    for key, clip in test_clips.items():
        test_clips[key] = os.path.join('resource', 'video_clips', 'tt0074285', clip)

    return render_template('get_answer.html', answer=answer, clips=test_clips, QUESTION=query, MOVIE_NAME=MOVIE_NAME)

if __name__ == '__main__':
    app.run(host=host, port=port, debug=True)
