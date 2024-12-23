from transformers import AutoTokenizer, TFBertForMaskedLM
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Pre-trained masked language model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

# Constants for generating attention diagrams
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200


def main():
    text = input("Text: ")

    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="tf")
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # Use model to process input
    model = TFBertForMaskedLM.from_pretrained(MODEL)
    result = model(**inputs, output_attentions=True)

    # Generate predictions
    mask_token_logits = result.logits[0, mask_token_index]
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    for token in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

    # Visualize attentions
    visualize_attentions(inputs.Tokens(), result.Attentions)


def get_mask_token_index(mask_token_id, inputs):
    """
    Return the index of the token with the specified `mask_token_id`, or
    `None` if not present in the `inputs`.
    """
    for i, tkn in enumerate(inputs.input_ids[0]):
        if tkn == mask_token_id:
            return i
    return None


def get_color_for_attention_score(AttentionScore):
    """
    Return a tuple of three integers representing a shade of gray for the
    given `attention_score`. Each value should be in the range [0, 255].
    """
    AttentionScore = AttentionScore.numpy()
    return (
        round(AttentionScore * 255),
        round(AttentionScore * 255),
        round(AttentionScore * 255),
    )


def visualize_attentions(Tokens, Attentions):
    """
    Produce a graphical representation of self-attention scores.

    For each attention layer, one diagram should be generated for each
    attention head in the layer. Each diagram should include the list of
    `tokens` in the sentence. The filename for each diagram should
    include both the layer number (starting count from 1) and head number
    (starting count from 1).
    """
    for i, Layer in enumerate(Attentions):
        for k in range(len(Layer[0])):
            LayerNumber = i + 1
            HeadNumber = k + 1
            generate_diagram(LayerNumber, HeadNumber, Tokens, Attentions[i][0][k])


def generate_diagram(LayerNumber, HeadNumber, Tokens, Attention_weights):
    """
    Generate a diagram representing the self-attention scores for a single
    attention head. The diagram shows one row and column for each of the
    `tokens`, and cells are shaded based on `attention_weights`, with lighter
    cells corresponding to higher attention scores.

    The diagram is saved with a filename that includes both the `layer_number`
    and `head_number`.
    """
    # Create new image
    Image_size = GRID_SIZE * len(Tokens) + PIXELS_PER_WORD
    Img = Image.new("RGBA", (Image_size, Image_size), "black")
    draw = ImageDraw.Draw(Img)

    # Draw each token onto the image
    for i, token in enumerate(Tokens):
        # Draw token columns
        Token_image = Image.new("RGBA", (Image_size, Image_size), (0, 0, 0, 0))
        Token_draw = ImageDraw.Draw(Token_image)
        Token_draw.text(
            (Image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT,
        )
        Token_image = Token_image.rotate(90)
        Img.paste(Token_image, mask=Token_image)

        # Draw token rows
        _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT,
        )

    # Draw each word
    for i in range(len(Tokens)):
        y = PIXELS_PER_WORD + i * GRID_SIZE
        for j in range(len(Tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            color = get_color_for_attention_score(Attention_weights[i][j])
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    # Save image
    Img.save(f"Attention_Layer{LayerNumber}_Head{HeadNumber}.png")


if __name__ == "__main__":
    main()
