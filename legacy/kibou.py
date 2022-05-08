B0 = next(iter(mytrainer.get_train_dataloader()))
def on_device(inputs, device, tolabel=False):
    input_ids = inputs["input_ids"]
    to_return = dict(
        input_ids=input_ids.to(device),
        word_length_tensor=inputs.get("word_length_tensor", None).to(device),
        attention_mask=inputs.get("attention_mask", None).to(device),
    )
    if tolabel:
        labels = input_ids.masked_fill(input_ids==503, -100).to(device)
        return {"labels": labels, **to_return}
    else:
        return to_return
B = on_device(B0, "cuda")
import torch
with torch.no_grad():
    L = AEModel(**B)
    TFOutput = L.logits.argmax(-1)
    OOO = AEModel.generate(B)
    
AEModel.generate
AEModel.generate(B['input_ids'])  # TOOO SHOTJRT! TODO
L.keys()
L.loss
hist

####################################3
####################################3
####################################3
####################################3
####################################3
####################################3
####################################3
####################################3
class C: pass
self = C()

deviced_dataset = UnitDataset(df[:20], bart_tokenizer)

from torch.utils.data import DataLoader
if "real":
    MAXUNITLEN = 1024
    MAXTEXTLEN = 512
else:
    MAXUNITLEN = 200
    MAXTEXTLEN = 28
self.valset = deviced_dataset
self.batch_size = 3
self.tokenizer = bart_tokenizer

deviced_dataloader = DataLoader(
    dataset=self.valset, 
    batch_size=self.batch_size,
    shuffle=False,
    num_workers=self.valset.num_workers,
    collate_fn=UnitDataset.tokenized_collate_fn(
        self.tokenizer, 
        padding_value=503, 
        max_unit_length=MAXUNITLEN, 
        max_text_length=MAXTEXTLEN,
    ),
)


from transformers import BartConfig

checkpoint_name = 'facebook/bart-base'
S = advanced_load_pretrained(
    checkpoint_name=checkpoint_name,
    model_class=SentBartForConditionalGeneration, 
    config_class=BartConfig, 
    # new_config_options...
    max_position_embeddings=1024, 
    vocab_size=504).cuda()
    
####################################3
import torch
from tqdm import tqdm
OO = torch.optim.AdamW(S.parameters(), lr=3e-4)
# OO = torch.optim.AdamW(S.parameters(), lr=1e-4)
# OO = torch.optim.AdamW(S.parameters(), lr=5e-5)
# OO = torch.optim.AdamW(S.parameters(), lr=2e-5)


for E in range(50):
    T = 0.
    for U in tqdm(deviced_dataloader):
        OO.zero_grad()
        Lab = U["input_ids"].masked_fill(U["input_ids"]==503, -100)

        V = S(
            input_ids=U["input_ids"].cuda(),
            word_length_tensor=U["word_length_tensor"].cuda(),
            attention_mask=U["attention_mask"].cuda(),
            labels=Lab.cuda(),
        )
        # V.logits.argmax(-1)
        # print('\n'.join(list(V.keys())))
        LL = (V.loss)
        # print(LL.item())
        T += (LL.item())
        LL.backward()
        OO.step()
    print('\033[01;31m', T, '\033[0m')

####################################3
assert S.get_encoder().config.hidden_size > 0

# TODO: bos, eos ...