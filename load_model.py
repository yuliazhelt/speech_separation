import gdown

id = "1DMfYcDYZJAasEE0p-UB4JMJEF7SznGel"
output = "model_best.pth"
gdown.download(id=id, output=output, quiet=False)