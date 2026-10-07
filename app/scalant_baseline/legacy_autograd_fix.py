"""Preserve ScalAnt's released forward values without mutating saved layer inputs.

The released decoder layers add positions in place to both the query and the
shared memory.  Copying the inputs *between* layers keeps the cumulative
position additions in the forward pass while giving autograd a stable tensor
from each preceding layer.  This is an external compatibility patch: the
upstream submodule stays byte-for-byte unchanged.
"""

from __future__ import annotations


def install_clone_per_layer_fix() -> None:
    from scalant.models.clam import CLAM
    from scalant.models.decoder import QueryDecoder

    def clam_forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                     tgt_key_padding_mask=None, memory_key_padding_mask=None,
                     pos=None, query_pos=None):
        output = tgt
        for layer in self.layers:
            # with_pos_embed mutates its arguments. The next layer must receive
            # the previous layer's *values*, but on fresh storage.
            output = output.clone()
            memory = memory.clone()
            output = layer(
                output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos, query_pos=query_pos,
            )
        return self.norm(output)

    def decoder_forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                        tgt_key_padding_mask=None, memory_key_padding_mask=None,
                        pos=None, query_pos=None):
        output = tgt
        intermediate = []
        if self.return_intermediate and self.tgt_type == "data_dependent":
            intermediate.append(self.norm(output))
        for layer in self.layers:
            output = output.clone()
            memory = memory.clone()
            output = layer(
                output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos, query_pos=query_pos,
                past_embed=self.past_embed, future_embed=self.future_embed,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        output = self.norm(output)
        return intermediate if self.return_intermediate else output

    CLAM._forward = clam_forward
    QueryDecoder._forward = decoder_forward
