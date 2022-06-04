import imghdr, os, PIL
from PIL import Image
def resize_height(img_path, new_height = 800):
    try:
        img = Image.open(img_path)
        w, h = img.size
        if h > new_height:
            new_width = round(new_height * w / h)
            img = img.resize((new_width, new_height))
            img.save(img_path)
        return None
    except:
        return [imghdr.what(img_path), img_path]
def resize_width(img_path, new_width = 800):
    try:
        img = Image.open(img_path)
        w, h = img.size
        if w > new_width:
            new_height = round(new_width * h / w)
            img = img.resize((new_width, new_height))
            img.save(img_path)
        return None
    except:
        return [imghdr.what(img_path), img_path]
def resize_image(img_path, max_dimm = 800):
    try:
        img = Image.open(img_path)
        curr_max_dimm = max(img.size)
        if curr_max_dimm > max_dimm:
            w, h = img.size
            ratio = (max_dimm / curr_max_dimm)
            n_w = round(w * ratio)
            n_h = round(h * ratio)
            img = img.resize((n_w, n_h))
            img.save(img_path)
        return None
    except:
        return [imghdr.what(img_path), img_path]