import torch

class GRU_aucdio(torch.nn.Module):
    def __init__(self, audio_output_size=768):
        super().__init__()
        self.audio_output_size = audio_output_size
        self.bigru = torch.nn.GRU(input_size=5, hidden_size=audio_output_size, num_layers=2, bias=True,
                                  batch_first=True, bidirectional=False)
        self.acu_linear = torch.nn.Linear(375, 80, bias=True)


    def forward(self, aucdio_out):

        gru_output_sequence, _ = self.bigru(aucdio_out)
        gru_output_sequence = gru_output_sequence.transpose(1, 2)
        gru_output_sequence = self.acu_linear(gru_output_sequence)
        gru_output_sequence = gru_output_sequence.transpose(1, 2)

        return gru_output_sequence

class GRU_visual(torch.nn.Module):
    def __init__(self, audio_output_size=768):
        super().__init__()
        self.audio_output_size = audio_output_size
        self.bigru = torch.nn.GRU(input_size=20, hidden_size=audio_output_size, num_layers=2, bias=True,
                                  batch_first=True, bidirectional=False)
        self.acu_linear = torch.nn.Linear(500, 80, bias=True)

    def forward(self, fp_out):

        gru_output_sequence, _ = self.bigru(fp_out)
        gru_output_sequence = gru_output_sequence.transpose(1, 2)
        gru_output_sequence = self.acu_linear(gru_output_sequence)
        gru_output_sequence = gru_output_sequence.transpose(1, 2)

        return gru_output_sequence

