# MRC-models
The latest model of Machine Reader Comprehension, include match-LSTM, Bi-DAF, R-net, Mnemonic Reader
LR=0.001	Rough-L	Bleu-4
match-LSTM+char_CNN_embedding+match_pointer+fusion	0.420008771	0.261740285
match-LSTM+char_RNN_embedding+match_pointer+fusion	0.445138158	0.303054566
match-LSTM+char_CNN_embedding+r_net_pointer+fusion	0.452427949	0.291018282
match-LSTM+char_RNN_embedding+r_net_pointer+fusion	0.431437641	0.313959133
match-LSTM+char_CNN_embedding+r_net_pointer+fusion+highway_network	0.410188406	0.183880396
match-LSTM+char_RNN_embedding+r_net_pointer+fusion+highway_network	0.442507766	0.271002596
		
Bi-DAF+char_CNN_embedding+match_pointer+highway_network+fusion	0.44869017	0.290897368
Bi-DAF+char_CNN_embedding+r_net_pointer+highway_network+fusion	0.458529946	0.323599176
Bi-DAF+char_CNN_embedding+r_net_pointer	0.36925992	0.204449311
Bi-DAF+char_RNN_embedding+r_net_pointer+highway_network	0.212263432	0.07437824
Bi-DAF+char_CNN_embedding+MRC_pointer+highway_network+fusion	0.444801456	0.28792319
		
R-NET+char_RNN_embedding+r_net_pointer	0.422188195	0.240846086
R-NET+char_CNN_embedding+r_net_pointer	0.441353237	0.292194921
R-NET+char_CNN_embedding+r_net_pointer+highway_network	0.451170429	0.302083755
R-NET+char_CNN_embedding+r_net_pointer+highway_network+fusion	0.431182324	0.247489018
R-NET+char_CNN_embedding+match_pointer+highway_network	0.412151921	0.220152777
R-NET+char_CNN_embedding+MRC_pointer+highway_network	0.448774574	0.274030286
		
Mnemonic Reader+char_RNN_embedding+MRC_pointer+fusion	0.451744448	0.293358019
Mnemonic Reader+char_CNN_embedding+MRC_pointer+fusion	0.45339174	0.298865684
Mnemonic Reader+char_CNN_embedding+MRC_pointer	0.457618586	0.323924252
Mnemonic Reader+char_CNN_embedding+r_net_pointer	0.457286537	0.327276166
Mnemonic Reader+char_CNN_embedding+MRC_pointer+highway_network	0.468924931	0.308296227
Mnemonic Reader+char_RNN_embedding+MRC_pointer+highway_network	0.459134862	0.289311221
Mnemonic Reader+char_CNN_embedding+MRC_pointer+highway_network+fusion	0.445258024	0.280500902
Mnemonic Reader+char_CNN_embedding+r_net_pointer+highway_network	0.432951398	0.277616646
