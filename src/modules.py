from transformers import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from torch import nn
import sys


class ClsModel(BertPreTrainedModel):

    def __init__(self, num_labels, ptm_name, config=None):
        super().__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(ptm_name)

        if config is not None:
            hidden_dropout_prob = config.hidden_dropout_prob
            hidden_size = config.hidden_size
        else:
            hidden_dropout_prob = self.bert.config.hidden_dropout_prob
            hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.activation = nn.Tanh()
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None):
        bert_outputs = self.bert(input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 position_ids=position_ids,
                                 head_mask=head_mask,
                                 inputs_embeds=inputs_embeds)
        last_hidden_states = bert_outputs[0]
        trans_states = self.dense(last_hidden_states)
        trans_states = self.activation(trans_states)
        trans_states = self.dropout(trans_states)
        logits = self.classifier(trans_states)

        return logits
