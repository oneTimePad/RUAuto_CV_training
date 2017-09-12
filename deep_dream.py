import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import inception
import os
from PIL import Image
import numpy as np
from io import BytesIO
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_path',None,'path to input image')
tf.app.flags.DEFINE_string('model_ckpt','inception','Checkpoint directory')
tf.app.flags.DEFINE_integer('num_iter',5,'number of algorithm iterations')



image = tf.placeholder(shape=(None,None,None,3),name='image',dtype=tf.float32)
scaled = tf.multiply(image,2.0/255.0)
scaled = tf.subtract(scaled,1.0)
step = tf.contrib.framework.get_or_create_global_step()
with slim.arg_scope(inception.inception_v3_arg_scope()):
	_,end_points = inception.inception_v3(
					scaled,num_classes=1001,is_training=False)
	global_step = tf.contrib.framework.get_or_create_global_step()
	vars_to_restore = slim.get_variables_to_restore(exclude=['InceptionV3/Logits','InceptionV3/AuxLogits'])


def showaray(a,fmt='jpeg'):
	a = np.uint8(np.clip(a,0,1)*255)
	f = open('./test.jpg','wb+')
	Image.fromarray(a).save(f,fmt)
	f.close()

def norm(a,s=0.1):
	return (a-a.mean())/max(a.std(),1e-4)*s +.5

def tffunc(*argtypes):
	#from TF tutorial, allows TF graph operations to act as regular functions
	placeholders = list(map(tf.placeholder,argtypes))
	def wrap(f):
		out = f(*placeholders)
		def wrapper(*args,**kw):
			return out.eval(dict(zip(placeholders,args)),session=kw.get('session'))
		return wrapper
	return wrap

def resize(img,size):
	img = tf.expand_dims(img,0)
	return tf.image.resize_bilinear(img,size)[0,:,:,:]
resize = tffunc(np.float32,np.int32)(resize)

def calc_grad_tiled(img,t_grad,tile_size=299):
	sz = tile_size
	h, w = img.shape[:2]
	sx,sy = np.random.randint(sz,size=2)
	img_shift = np.roll(np.roll(img,sx,1),sy,0)
	grad = np.zeros_like(img)

	for y in range(0,max(h-sz//2,sz),sz):
		for x in range(0,max(w-sz//2,sz),sz):
			sub = img_shift[y:y+sz,x:x+sz]
			g = sess.run(t_grad,{image:[sub]})[0]
			grad[y:y+sz,x:x+sz] = g
	return np.roll(np.roll(grad,-sx,1),-sy,0)
def render_deepdream(t_obj,img,iter_n=10,step = 1.5, octave_n=4, octave_scale=1.4):
	t_score = tf.reduce_mean(t_obj)
	t_grad = tf.gradients(t_score,image)[0]

	octaves = []
	for i in range(octave_n-1):
		hw = img.shape[:2]
		lo = resize(img,np.int32(np.float32(hw)/octave_scale))
		hi = img - resize(lo,hw)
		img = lo
		octaves.append(hi)

	for octave in range(octave_n):
		if octave > 0:
			hi = octaves[-octave]
			img = resize(img,hi.shape[:2]) + hi
		for i in range(iter_n):
			g = calc_grad_tiled(img,t_grad)
			img += g*(step/(np.abs(g).mean()+1e-7))
			print('.',end = '')
	return img

def naive(t_obj,img):
	t_score = tf.reduce_mean(t_obj)
	t_grad = tf.gradients(t_score,image)[0]
	step = 1.0
	for i in range(FLAGS.num_iter):
		g,score = sess.run([t_grad,t_score],{image:[img]})
		g = g[0]
		g /= g.std() + 1e-8
		img  += g*step
		print(score)
	return img
saver = tf.train.Saver(var_list=vars_to_restore)
with tf.Session() as sess:
    print(FLAGS.image_path)
    print(os.path.join(FLAGS.model_ckpt,'inception_v3.ckpt'))
    saver.restore(sess,os.path.join(FLAGS.model_ckpt,'inception_v3.ckpt'))
    img = Image.open(FLAGS.image_path)
    img0 = np.float32(img)
    img_cpy = img0.copy()
    print(end_points.keys())

    tensor = end_points['Mixed_6b']

    t_obj = tf.square(tensor)
    #t_obj  = tensor[:,:,:,100]
    img = render_deepdream(t_obj,img_cpy,step=2,iter_n=10)

    showaray(img/255.0)
