from flask import Flask, request, redirect, render_template
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
import torch
import torchvision
import torch.nn.functional as F

from utils import convert_img
import cv2 

app = Flask(__name__)
# APP_ROOT = os.path.dirname(os.path.abspath(__file__))
# IMAGE_UPLOADS = os.path.join(APP_ROOT, 'static/')
app.config["IMAGE_UPLOADS"] = 'static'
app.config["ALLOWED_IMAGE_EXTENTIONS"] = ["JPEG", "JPG", "PNG"]
# basedir = os.path.abspath(os.path.dirname(__file__))
# print(basedir)
model = torch.load('network.pth')
model.eval()



images = []
original_images = []
for im in os.listdir('images'):

	im_path = os.path.join('images', im)
	original_images.append(Image.open(im_path))
	im = convert_img(im_path)
	images.append(im)

def allowed_image(filename):

	if not "." in filename:
		return False

	ext = filename.rsplit(".", 1)[1]

	if ext.upper() in app.config["ALLOWED_IMAGE_EXTENTIONS"]:
		return True
	else:
		return False

@app.route("/", methods=["GET", "POST"])
def upload_image():

	if request.method == 'POST':
		if request.files:
			image = request.files["file"]

			if image.filename == "":
				return redirect(request.url)

			if allowed_image(image.filename):
				
				filename = secure_filename(image.filename)
				image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

				return redirect(f'/showing-image/{filename}')
			else:
				return redirect(request.url)

	return render_template('upload_images.html')



@app.route("/showing-image/<image_name>", methods=["GET", "POST"])
def showing_image(image_name):
	min_val = 100
	dists = []
	if request.method == 'POST':
		image_path = os.path.join(app.config["IMAGE_UPLOADS"], image_name)

		image = Image.open(image_path)
		img = convert_img(image_path)

		for i, im in enumerate(images):
			out1, out2 = model(img.cuda(), im.cuda())
			euc_dist = F.pairwise_distance(out1, out2)
			dists.append(euc_dist.item())

			org_im = original_images[i]
			res = Image.new('RGB', (image.width + org_im.width, max(image.height, org_im.height)))
			res.paste(image, (0, 0))
			res.paste(org_im, (image.width, 0))
			res = np.array(res)
			cv2.putText(res, f'{euc_dist.item():.2}', (10, 100), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3, color=(0, 255, 0),thickness=3)
			res = Image.fromarray(res)
			res_name = f'res_{i}.jpg'
			res.save(os.path.join(app.config["IMAGE_UPLOADS"], 'res/', res_name))
			all_res = [f'res/{r}' for r in os.listdir(os.path.join(app.config["IMAGE_UPLOADS"], 'res/'))]

			if min_val > euc_dist.item():
				min_val = euc_dist.item() 
				concatened = torch.cat((img, im), 0)
				concatened = torchvision.utils.make_grid(concatened)
				
				npimg = concatened.numpy()
				npimg = npimg * 255.
				npimg = npimg.astype(np.uint8)
				npimg = np.transpose(npimg, (1, 2, 0))
				
				sh = npimg.shape
				result_img = Image.fromarray(npimg)

				result_name = 'concatenated.png'
				result_img.save(os.path.join(app.config["IMAGE_UPLOADS"], result_name))
				

		return render_template('showing_result.html', image_name=result_name, res_img=all_res, min_val=round(min_val, 2), dists=all_res)
	
	return render_template('showing_image.html', value=image_name)