from transformers import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from torch import nn
import sys


class OurModel(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.token_cls = BertOnlyMLMHead(config)

        #         self.dropout = nn.Dropout(hidden_dropout_prob)
        #         self.dense = nn.Linear(hidden_size, hidden_size)

        self.init_weights()

    def _init_vars(self, num_labels):
        self.num_labels = num_labels
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(768, num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, pred_mode='cls'):
        bert_outputs = self.bert(input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 position_ids=position_ids,
                                 head_mask=head_mask,
                                 inputs_embeds=inputs_embeds)

        if pred_mode == 'cls':
            out = bert_outputs[0][:, 0, :]
            # trans_states = self.dense(last_hidden_states)
            out = self.activation(out)
            # trans_states = self.dropout(trans_states)
            logits = self.classifier(out)
        elif pred_mode == 'mlm':
            out = bert_outputs[0]
            logits = self.token_cls(out)
        else:
            raise ValueError('Only support "cls" and "mlm" mode.')
        return logits
