
import os
import sys
import uuid
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, render_template, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField
from wtforms.validators import InputRequired, NumberRange
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR / "pytorch-AdaIN"

VGG_PATH = BASE_DIR / "weights_adain_official" / "vgg_normalised.pth"
DECODER_PATH = BASE_DIR / "experiment" / "adain_finetune20" / "decoder_epoch_20.pth"

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
EXAMPLES_FOLDER = BASE_DIR / "example"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

EXAMPLE_CONTENT = "Srikanth.jpeg"
EXAMPLE_STYLE = "style.jpg"
EXAMPLE_RESULT = "Result.png"

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get(
    "FLASK_SECRET_KEY", "change-this-development-key"
)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

Bootstrap(app)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for path in [
    REPO_DIR / "net.py",
    REPO_DIR / "function.py",
    VGG_PATH,
    DECODER_PATH,
]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")

for filename in [EXAMPLE_CONTENT, EXAMPLE_STYLE, EXAMPLE_RESULT]:
    if not (EXAMPLES_FOLDER / filename).is_file():
        raise FileNotFoundError(
            f"Example image not found: {EXAMPLES_FOLDER / filename}"
        )

sys.path.insert(0, str(REPO_DIR))

import net
from function import adaptive_instance_normalization as adain

vgg_weights = torch.load(
    VGG_PATH,
    map_location="cpu",
    weights_only=True,
)
net.vgg.load_state_dict(vgg_weights, strict=True)

vgg = nn.Sequential(*list(net.vgg.children())[:31])

decoder_weights = torch.load(
    DECODER_PATH,
    map_location="cpu",
    weights_only=True,
)
net.decoder.load_state_dict(decoder_weights, strict=True)

model = net.Net(vgg, net.decoder).to(device)
model.eval()

print("AdaIN VGG loaded successfully.")
print("Fine-tuned decoder loaded successfully.")
print("Device:", device)


class UploadForm(FlaskForm):
    content = FileField("Content Image", validators=[InputRequired()])
    style = FileField("Style Image", validators=[InputRequired()])
    alpha = FloatField(
        "Alpha",
        default=1.0,
        validators=[
            InputRequired(),
            NumberRange(min=0.0, max=1.0),
        ],
    )
    submit = SubmitField("Transfer Style")


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def style_transfer(content_image, style_image, alpha=1.0):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    content = transform(content_image).unsqueeze(0).to(device)
    style = transform(style_image).unsqueeze(0).to(device)

    alpha = max(0.0, min(1.0, float(alpha)))

    with torch.inference_mode():
        content_features = model.encode(content)
        style_features = model.encode_with_intermediate(style)

        target_features = adain(
            content_features,
            style_features[-1],
        )

        target_features = (
            alpha * target_features
            + (1.0 - alpha) * content_features
        )

        output = model.decoder(target_features)

    return output.clamp(0, 1)


def save_image(tensor, path):
    tensor = tensor.detach().cpu().squeeze(0).clamp(0, 1)
    image = transforms.ToPILImage()(tensor)
    image.save(path)


@app.route("/", methods=["GET", "POST"])
def index():
    form = UploadForm()
    result_image = None
    content_image = None
    style_image = None
    error = None

    if form.validate_on_submit():
        content_file = form.content.data
        style_file = form.style.data

        if not content_file or not style_file:
            error = "Please upload both content and style images."
        elif not allowed_file(content_file.filename):
            error = "Unsupported content image format."
        elif not allowed_file(style_file.filename):
            error = "Unsupported style image format."
        else:
            content_filename = (
                uuid.uuid4().hex[:12]
                + "_"
                + secure_filename(content_file.filename)
            )
            style_filename = (
                uuid.uuid4().hex[:12]
                + "_"
                + secure_filename(style_file.filename)
            )

            content_path = UPLOAD_FOLDER / content_filename
            style_path = UPLOAD_FOLDER / style_filename

            try:
                content_file.save(str(content_path))
                style_file.save(str(style_path))

                with Image.open(content_path) as image:
                    content_pil = image.convert("RGB")

                with Image.open(style_path) as image:
                    style_pil = image.convert("RGB")

                output = style_transfer(
                    content_pil,
                    style_pil,
                    form.alpha.data,
                )

                result_filename = f"stylized_{uuid.uuid4().hex[:12]}.png"
                result_path = UPLOAD_FOLDER / result_filename

                save_image(output, result_path)

                content_image = content_filename
                style_image = style_filename
                result_image = result_filename

            except Exception:
                app.logger.exception("Style transfer failed")
                error = "Style transfer failed. Check the terminal for details."

    elif __import__("flask").request.method == "POST":
        error = "Please upload both images and enter an alpha value from 0 to 1."

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_image,
        style_image=style_image,
        error=error,
        example_content=EXAMPLE_CONTENT,
        example_style=EXAMPLE_STYLE,
        example_result=EXAMPLE_RESULT,
    )


@app.route("/uploads/<path:filename>")
def send_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/example/<path:filename>")
def send_example(filename):
    return send_from_directory(EXAMPLES_FOLDER, filename)


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )
