import tensorflow as tf
from flask import Flask, render_template, request
import os
import json
import demo

from demo.main import prepare_data
from demo.model import MemN2N

host = '0.0.0.0'
port = 8999


movie_info_path = '/data/movieQA/MovieQA_benchmark/data/movies.json'
with open(movie_info_path, 'r') as f:
    movie_info = json.load(f)

qa_info_path = '/data/movieQA/MovieQA_benchmark/data/qa.json'
with open(qa_info_path, 'r') as f:
    qa_info = json.load(f)

global qid_info
qid_info = {}
for info in qa_info:
    qid_info[info['qid']] = info

global movie_imdb
movie_imdb = {}
for info in movie_info:
    movie_imdb[info['name']] = info['imdb_key']

global MOVIE_NAME
MOVIE_NAME = 'Movie Name'

ROOT_DIR = os.getcwd()
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates/resource')
app = Flask(__name__, static_folder=ASSETS_DIR)

FLAGS = demo.main.FLAGS

global sess
global model

sess = tf.Session(config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.95),
    device_count={'GPU': 3}))
model = MemN2N(FLAGS, sess)
model.build_model(mode='inference', embedding_method='word2vec')


@app.route('/')
def hello_name():
    return render_template('index.html', MOVIE_NAME=MOVIE_NAME)


@app.route("/result", methods=['POST', 'GET'])
def result():
    global MOVIE_NAME
    if request.method == 'POST':
        result = request.form
        MOVIE_NAME = result['movie_name']
        if MOVIE_NAME not in movie_imdb.keys():
            MOVIE_NAME += ' NOT FOUND'
            return render_template('index.html', MOVIE_NAME=MOVIE_NAME, clip_path='')
        movie_path = os.path.join('video_clips', movie_imdb[MOVIE_NAME])
        try:
            video_clips = [clip for clip in os.listdir(os.path.join(ASSETS_DIR, movie_path))]
            clip_path = os.path.join('video_clips', movie_imdb[MOVIE_NAME], video_clips[0])
            return render_template('load_movie.html', clip_path=os.path.join('resource', clip_path), MOVIE_NAME=MOVIE_NAME)
        except:
            MOVIE_NAME += ' is not supported movie.'
            return render_template('load_movie.html', MOVIE_NAME=MOVIE_NAME, clip_path='')
    return render_template('load_movie.html')


@app.route("/question", methods=['POST', 'GET'])
def question():
    global MOVIE_NAME
    result = request.form
    query = result['question_text']

    qa_info, s, q, a = prepare_data(query, movie_imdb[MOVIE_NAME])
    data = s, q, a
    print qa_info
    answer_index = model.inference(data)
    clips = qa_info.video_clips
    answer = qa_info.answers[answer_index[0]]

    new_clips = []
    for i, clip in enumerate(clips):
        new_clips.append(os.path.join('resource', 'video_clips', movie_imdb[MOVIE_NAME], clip))

    return render_template('get_answer.html', enumerate=enumerate, answer=answer, clips=new_clips, QUESTION=query, MOVIE_NAME=MOVIE_NAME)

if __name__ == '__main__':
    app.run(host=host, port=port)
