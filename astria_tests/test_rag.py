import copy

from test_infer import pipe, BASE_PROMPT, run_images, IMG_POSE, FLUX_LORA, JsonObj, RAG_FluxPipeline, FLUX_LORA_SHOE, \
    FLUX_LORA_MAN_MARCO, FLUX_LORA_MAN, FLUX_CARTOON


def test_regional():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional=True,
    )
    assert prompt.use_regional is True
    prompt.text = "a dog inbetween two vases full of flowers. the flowers on the left are white lillies, the flowers on the right are roses. the dog is a pembroke welsh corgi. above the corgi, there are balloons flying that say \"happy birthday\""
    run_images(prompt)
    assert isinstance(pipe.last_pipe, RAG_FluxPipeline)

def test_regional_lora():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional=True,
    )
    prompt.tunes=[FLUX_LORA]
    prompt.text = f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman inbetween two vases full of flowers. the flowers on the left are white lillies, the flowers on the right are roses. above the {FLUX_LORA.train_token} woman, there are balloons flying that say \"happy birthday\""
    run_images(prompt)
    assert isinstance(pipe.last_pipe, RAG_FluxPipeline)

def test_regional_two_lora():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional=True,
    )
    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers standing beside <lora:{FLUX_LORA_MAN_MARCO.id}:1> {FLUX_LORA_MAN_MARCO.train_token} man"
    prompt.tunes=[FLUX_LORA, FLUX_LORA_MAN_MARCO]
    run_images(prompt)
    assert isinstance(pipe.last_pipe, RAG_FluxPipeline)

def test_regional_one_lora_famous_person():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional=True,
    )
    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers standing beside donald trump in the white house"
    prompt.tunes=[FLUX_LORA]
    run_images(prompt)
    assert isinstance(pipe.last_pipe, RAG_FluxPipeline)

def test_regional_two_lora_person_and_object():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional=True,
    )
    prompt.text=f"<lora:{FLUX_LORA_MAN.id}:1> {FLUX_LORA_MAN.train_token} man with <lora:{FLUX_LORA_SHOE.id}:1> {FLUX_LORA_SHOE.train_token} shoe on and prominently visible, walking on a gravel path in a forest"
    prompt.tunes=[FLUX_LORA_MAN, FLUX_LORA_SHOE]
    run_images(prompt)
    assert isinstance(pipe.last_pipe, RAG_FluxPipeline)

def test_regional_one_lora_person_and_cartoon_lora():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional=True,
    )
    prompt.text=f"<lora:{FLUX_LORA_MAN.id}:1> {FLUX_LORA_MAN.train_token} man, real photograph portrait holding <lora:{FLUX_CARTOON.id}:1> {FLUX_CARTOON.train_token} sloth, white t-shirt, white background, professional headshot with cartoon sloth character. The top half of the image is the man's shoulder and face, which the bottom half of the image is the 2d illustration being held in his arms. Compositing cinematography example, cartoon overlaid on photograph"
    prompt.tunes=[FLUX_LORA_MAN, FLUX_CARTOON]
    run_images(prompt)
    assert isinstance(pipe.last_pipe, RAG_FluxPipeline)
