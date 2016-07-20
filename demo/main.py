from model import MemN2N
import tensorflow as tf
import data_unit
import sys
import numpy as np
sys.path.append('/data/movieQA/MovieQA_benchmark')
sys.path.append('/data/movieQA/skip-thoughts')
import skipthoughts as skip
import data_loader as MovieQA
import gensim_w2v
import configs


mqa = MovieQA.DataLoader()
story, qa = mqa.get_story_qa_data('full', 'split_plot')
skip_model = skip.load_model()
INF = 987654321

def prepare_data(query, imdb_key):
  query_representation = skip.encode(skip_model, [query])
  candidate_qa = [QAInfo for QAInfo in qa if QAInfo.imdb_key == imdb_key]
  skip_encode = [skip.encode(skip_model, [QAInfo.question.lower()]) for QAInfo in candidate_qa]
  similarity = [(np.inner(query_representation, rep)[0][0], i) for i, rep in enumerate(skip_encode)]
  similarity.sort(reverse=True)
  most_similar = [candidate_qa[i] for score, i in similarity[:1]]

  retrieved_question = most_similar[0].question
  retrieved_answer = most_similar[0].answers
  retrieved_story = story[imdb_key]

  q_embed = np.array(gensim_w2v.encode_w2v_gensim(retrieved_question))
  a_embed = np.array([gensim_w2v.encode_w2v_gensim(a) for a in retrieved_answer])
  s_embed = np.zeros((1, 60, 300))
  s_embed[:,:len(retrieved_story)] = \
      np.reshape(np.array([gensim_w2v.encode_w2v_gensim(s) for s in retrieved_story]),\
                 (1,len(retrieved_story), 300))

  s_embed = np.reshape(s_embed, (1, 60, 300))
  q_embed = np.reshape(q_embed, (1, 1, 300))
  a_embed = np.reshape(a_embed, (1, 5, 300))

  return most_similar[0], s_embed, q_embed, a_embed

flags = configs.tf_flag()
FLAGS = flags.FLAGS
print(FLAGS.__flags.items())
def main(_):
  with tf.Session(config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.95),
    device_count={'GPU': 3})) as sess:
    model = MemN2N(FLAGS, sess)
    model.build_model(mode='inference', embedding_method='word2vec')

    qa_info, s, q, a = prepare_data('what is the name of the eaman', 'tt0147800')
    print qa_info.qid
    data = s, q, a
    answer_index = model.inference(data)
    print 'answer_index >> ', answer_index[0]

if __name__ == '__main__':
  tf.app.run()
