import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable
import random as rn
import torchvision
import sys

def my_softmax(input, axis=1):
	trans_input = input.transpose(axis, 0).contiguous()
	soft_max_1d = F.softmax(trans_input)
	return soft_max_1d.transpose(axis, 0)

def get_offdiag_indices(num_nodes):
	ones = torch.ones(num_nodes, num_nodes)
	eye = torch.eye(num_nodes, num_nodes)
	offdiag_indices = (ones - eye).nonzero().t()
	offdiag_indices_ = offdiag_indices[0] * num_nodes + offdiag_indices[1]
	return offdiag_indices, offdiag_indices_

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
	y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
	if hard:
		shape = logits.size()
		_, k = y_soft.data.max(-1)
		y_hard = torch.zeros(*shape)
		if y_soft.is_cuda:
			y_hard = y_hard.cuda()
		y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
		y = Variable(y_hard - y_soft.data) + y_soft
	else:
		y = y_soft
	return y

def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
	gumbel_noise = sample_gumbel(logits.size(), eps=eps)
	if logits.is_cuda:
		gumbel_noise = gumbel_noise.cuda()
	y = logits + Variable(gumbel_noise)
	return my_softmax(y / tau, axis=-1)

def sample_gumbel(shape, eps=1e-10):
	uniform = torch.rand(shape).float()
	return - torch.log(eps - torch.log(uniform + eps))

def encode_onehot(labels):
	#converts the labels list to set in order to remove duplicate
    classes = set(labels)

    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

class MLP(nn.Module):

	def __init__(self, n_in, n_hid, n_out, do_prob=0., do_bn=True):
		super().__init__()

		self.fc1 = nn.Linear(n_in, n_hid)
		self.fc2 = nn.Linear(n_hid, n_out)
		self.bn = nn.BatchNorm1d(n_out)
		self.dropout = nn.Dropout(p=do_prob)
		self.do_bn = do_bn

		self.init_weights()

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight.data)
				m.bias.data.fill_(0.1)
			elif isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def batch_norm(self, inputs):
		x = inputs.view(inputs.size(0) * inputs.size(1), -1)
		x = self.bn(x)
		return x.view(inputs.size(0), inputs.size(1), -1)
		
	def forward(self, inputs):
		x = F.elu(self.fc1(inputs))
		x = self.dropout(x)
		x = F.elu(self.fc2(x))
		if self.do_bn:
			x = self.batch_norm(x)
		return x

class InteractionNet(nn.Module):

	def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
		#n_in = 150, n_hid = 128, n_out = 3, do_prob = 0.5, factor = True
		super().__init__()

		#MLP Class structure
		#fc1 = nn.Linear(n_in, n_hid)
		#fc2 = nn.Linear(n_hid, n_out)
		#bn = nn.BatchNorm1d(n_out)
		#dropout = nn.Dropout(p=do_prob)
		
		self.factor = factor
		self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)             #150, 128, 128, 0.5
		self.mlp2 = MLP(n_hid*2, n_hid, n_hid, do_prob)          #256, 128, 128, 0.5
		self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)            #128, 128, 128, 0.5
		self.mlp4 = MLP(n_hid*3, n_hid, n_hid, do_prob) if self.factor else MLP(n_hid*2, n_hid, n_hid, do_prob)             #384, 128, 128, 0.5
		self.fc_out = nn.Linear(n_hid, n_out)					 #128, 3

		self.init_weights()

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight.data)
				m.bias.data.fill_(0.1)

	def node2edge(self, x, rel_rec, rel_send):
		receivers = torch.matmul(rel_rec, x)
		senders = torch.matmul(rel_send, x)
		edges = torch.cat([receivers, senders], dim=2)
		return edges

	def edge2node(self, x, rel_rec, rel_send):
		incoming = torch.matmul(rel_rec.t(), x)
		nodes = incoming / incoming.size(1)
		return nodes

	def forward(self, inputs, rel_rec, rel_send):              # input: [2N, v, t, c] = [2N, 25, 50, 3]
		x = inputs.contiguous()
		x = x.view(inputs.size(0), inputs.size(1), -1)         # [2N, 25, 50, 3] -> [2N, 25, 50*3=150]
		x = self.mlp1(x)                                       # [2N, 25, 150] -> [2N, 25, n_hid=128] -> [2N, 25, n_out=128]
		x = self.node2edge(x, rel_rec, rel_send)               # [N, 25, 256] -> [N, 600, 256]|[N, 600, 256]=[N, 600, 512]

		x = self.mlp2(x)                                       # [N, 600, 512] -> [N, 600, n_hid=256] -> [N, 600, n_out=256]

		x_skip = x
		if self.factor:
			x = self.edge2node(x, rel_rec, rel_send)           # [N, 600, 256] -> [N, 25, 256]
			x = self.mlp3(x)                                   # [N, 25, 256] -> [N, 25, n_hid=256] -> [N, 25, n_out=256]
			x = self.node2edge(x, rel_rec, rel_send)           # [N, 25, 256] -> [N, 600, 256]|[N, 600, 256]=[N, 600, 512]
			x = torch.cat((x, x_skip), dim=2)                  # [N, 600, 512] -> [N, 600, 512]|[N, 600, 256]=[N, 600, 768]
			x = self.mlp4(x)                                   # [N, 600, 768] -> [N, 600, n_hid=256] -> [N, 600, n_out=256]
		else:
			x = self.mlp3(x)
			x = torch.cat((x, x_skip), dim=2)
			x = self.mlp4(x)
		return self.fc_out(x)                                  # [N, 600, 256] -> [N, 600, 3]

class InteractionDecoderRecurrent(nn.Module):

	def __init__(self, n_in_node, edge_types, n_hid, do_prob=0., skip_first=True):
		super().__init__()

		self.msg_fc1 = nn.ModuleList([nn.Linear(2 * n_hid, n_hid) for _ in range(edge_types)])
		self.msg_fc2 = nn.ModuleList([nn.Linear(n_hid, n_hid) for _ in range(edge_types)])
		self.msg_out_shape = n_hid
		self.skip_first_edge_type = skip_first

		self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
		self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
		self.hidden_n = nn.Linear(n_hid, n_hid, bias=False)

		self.input_r = nn.Linear(n_in_node, n_hid, bias=True)  # 3 x 256
		self.input_i = nn.Linear(n_in_node, n_hid, bias=True)
		self.input_n = nn.Linear(n_in_node, n_hid, bias=True)

		self.out_fc1 = nn.Linear(n_hid, n_hid)
		self.out_fc2 = nn.Linear(n_hid, n_hid)
		self.out_fc3 = nn.Linear(n_hid, n_in_node)

		self.dropout1 = nn.Dropout(p=do_prob)
		self.dropout2 = nn.Dropout(p=do_prob)
		self.dropout3 = nn.Dropout(p=do_prob)

	def single_step_forward(self, inputs, rel_rec, rel_send, rel_type, hidden):
		receivers = torch.matmul(rel_rec, hidden)
		senders = torch.matmul(rel_send, hidden)
		pre_msg = torch.cat([receivers, senders], dim=-1)
		all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1), self.msg_out_shape)
		gpu_id = rel_rec.get_device()
		all_msgs = all_msgs.cuda(gpu_id)
		if self.skip_first_edge_type:
			start_idx = 1
			norm = float(len(self.msg_fc2)) - 1.
		else:
			start_idx = 0
			norm = float(len(self.msg_fc2))
		for k in range(start_idx, len(self.msg_fc2)):
			msg = torch.tanh(self.msg_fc1[k](pre_msg))
			msg = self.dropout1(msg)
			msg = torch.tanh(self.msg_fc2[k](msg))
			msg = msg * rel_type[:, :, k:k + 1]
			all_msgs += msg / norm
		agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
		agg_msgs = agg_msgs.contiguous()/inputs.size(2)

		r = torch.sigmoid(self.input_r(inputs) + self.hidden_r(agg_msgs))
		i = torch.sigmoid(self.input_i(inputs) + self.hidden_i(agg_msgs))
		n = torch.tanh(self.input_n(inputs) + r * self.hidden_n(agg_msgs))

		hidden = (1-i)*n + i*hidden

		pred = self.dropout2(F.relu(self.out_fc1(hidden)))
		pred = self.dropout2(F.relu(self.out_fc2(pred)))
		pred = self.out_fc3(pred)
		pred = inputs + pred

		return pred, hidden

	def forward(self, data, rel_type, rel_rec, rel_send, pred_steps=1, 
		        burn_in=False, burn_in_steps=1, dynamic_graph=False,
		        encoder=None, temp=None):
		inputs = data.transpose(1, 2).contiguous()
		time_steps = inputs.size(1)
		hidden = torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape)
		gpu_id = rel_rec.get_device()
		hidden = hidden.cuda(gpu_id)
		pred_all = []
		for step in range(0, inputs.size(1) - 1):
			if not step % pred_steps:
				ins = inputs[:, step, :, :]
			else:
				ins = pred_all[step - 1]
			pred, hidden = self.single_step_forward(ins, rel_rec, rel_send, rel_type, hidden)
			pred_all.append(pred)
		preds = torch.stack(pred_all, dim=1)
		return preds.transpose(1, 2).contiguous()

class AdjacencyLearn(nn.Module):

	def __init__(self, n_in_enc, n_hid_enc, edge_types, n_in_dec, n_hid_dec, node_num=25, partial=False):
		"""
			n_in_enc: 150
  			n_hid_enc: 128
  			edge_types: 3
  			n_in_dec: 3
  			n_hid_dec: 128
  			node_num: 25								  
		"""
		super().__init__()

		self.encoder = InteractionNet(n_in=n_in_enc,                        # 150
			                          n_hid=n_hid_enc,                      # 128
			                          n_out=edge_types,                     # 3
			                          do_prob=0.5,
			                          factor=True)
		self.decoder = InteractionDecoderRecurrent(n_in_node=n_in_dec,      # 3
			                                       edge_types=edge_types,   # 3
			                                       n_hid=n_hid_dec,         # 128
			                                       do_prob=0.5,
			                                       skip_first=True)
		self.offdiag_indices, _ = get_offdiag_indices(node_num)

		self.partial = partial
		self.video_encoder = VideoContrastive()

		off_diag = np.ones([node_num, node_num])-np.eye(node_num, node_num)
		self.rel_rec = torch.FloatTensor(np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32))
		self.rel_send = torch.FloatTensor(np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32))
		self.dcy = 0.1

		self.init_weights()

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def forward(self, inputs): # [N, 3, 50, 25, 2]

		N, C, T, V, M = inputs.size()
		x = inputs.permute(0, 4, 3, 1, 2).contiguous() # [N, 2, 25, 3, 50]
		x = x.contiguous().view(N*M, V, C, T).permute(0,1,3,2)  # [2N, 25, 50, 3]

		gpu_id = x.get_device()
		rel_rec = self.rel_rec.cuda(gpu_id)
		rel_send = self.rel_send.cuda(gpu_id)

		loss_inter, loss_intra, loss_video, loss_tsn = torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda()

		if not self.partial:
			v1_inter, v1_inter_aug, v2_inter, v3_inter, v1_intra, v1_intra_aug, v2_intra, v3_intra, seg_q, seg_k, tsn_q, tsn_k, order_label = self.video_encoder.forward_distribute(x)

			v1_inter = self.encoder(v1_inter, rel_rec, rel_send)
			v1_inter_aug = self.encoder(v1_inter_aug, rel_rec, rel_send)
			v2_inter = self.encoder(v2_inter, rel_rec, rel_send)
			v3_inter = self.encoder(v3_inter, rel_rec, rel_send)

			v1_intra = self.encoder(v1_intra, rel_rec, rel_send)
			v1_intra_aug = self.encoder(v1_intra_aug, rel_rec, rel_send)
			v2_intra = self.encoder(v2_intra, rel_rec, rel_send)
			v3_intra = self.encoder(v3_intra, rel_rec, rel_send)

			seg_q = self.encoder(seg_q, rel_rec, rel_send)
			seg_k = self.encoder(seg_k, rel_rec, rel_send)

			tsn_q = self.encoder(tsn_q, rel_rec, rel_send)
			tsn_k = self.encoder(tsn_k, rel_rec, rel_send)

			v1_inter = torch.transpose(self.video_encoder.mlp_inter_encode_q(torch.transpose(v1_inter, 1, 2)), 1, 2)
			v1_inter_aug = torch.transpose(self.video_encoder.mlp_inter_encode_k(torch.transpose(v1_inter_aug, 1, 2)), 1, 2)
			v2_inter = torch.transpose(self.video_encoder.mlp_inter_encode_k(torch.transpose(v2_inter, 1, 2)), 1, 2)
			v3_inter = torch.transpose(self.video_encoder.mlp_inter_encode_k(torch.transpose(v3_inter, 1, 2)), 1, 2)

			v1_intra = torch.transpose(self.video_encoder.mlp_intra_encode_q(torch.transpose(v1_intra, 1, 2)), 1, 2)
			v1_intra_aug = torch.transpose(self.video_encoder.mlp_intra_encode_k(torch.transpose(v1_intra_aug, 1, 2)), 1, 2)
			v2_intra = torch.transpose(self.video_encoder.mlp_intra_encode_k(torch.transpose(v2_intra, 1, 2)), 1, 2)
			v3_intra = torch.transpose(self.video_encoder.mlp_intra_encode_k(torch.transpose(v3_intra, 1, 2)), 1, 2)

			seg_q = torch.transpose(self.video_encoder.mlp_segments_encode_q(torch.transpose(seg_q, 1, 2)), 1, 2)
			seg_k = torch.transpose(self.video_encoder.mlp_segments_encode_k(torch.transpose(seg_k, 1, 2)), 1, 2)

			
			tsn = torch.transpose(self.video_encoder.mlp_tsn_encode(torch.transpose(torch.cat([tsn_q, tsn_k], dim=1), 1, 2)), 1, 2)

			with torch.no_grad():
				self.video_encoder.update_ptr(2*N)

			loss_inter = self.video_encoder.contrast_inter(v1_inter, {v1_inter_aug, v2_inter, v3_inter})
			loss_intra = self.video_encoder.contrast_intra(v1_intra, v1_intra_aug, {v2_intra, v3_intra})
			loss_video = self.video_encoder.contrast_video(seg_q, seg_k)
			loss_tsn = self.video_encoder.contrast_tsn(tsn, order_label)

			with torch.no_grad():
				if self.training:
					self.video_encoder._dequeue_and_enqueue({v1_inter_aug, v2_inter, v3_inter}, seg_k)


		self.logits = self.encoder(x, rel_rec, rel_send)
		self.N, self.v, self.c = self.logits.size()
		self.edges = gumbel_softmax(self.logits, tau=0.5, hard=True)
		self.prob = my_softmax(self.logits, -1)

		self.outputs = self.decoder(x, self.edges, rel_rec, rel_send, burn_in=False, burn_in_steps=40)
		self.offdiag_indices = self.offdiag_indices.cuda(gpu_id)

		A_batch = []
		for i in range(self.N):
			A_types = []
			for j in range(1, self.c):
				A = torch.sparse.FloatTensor(self.offdiag_indices, self.edges[i,:,j], torch.Size([25, 25])).to_dense().cuda(gpu_id)
				A = A + torch.eye(25, 25).cuda(gpu_id)
				D = torch.sum(A, dim=0).squeeze().pow(-1)+1e-10
				D = torch.diag(D)
				A_ = torch.matmul(A, D)*self.dcy
				A_types.append(A_)
			A_types = torch.stack(A_types)
			A_batch.append(A_types)
		self.A_batch = torch.stack(A_batch).cuda(gpu_id) # [N, 2, 25, 25]

		return self.A_batch, self.prob, self.outputs, x, loss_inter, loss_intra, loss_video, loss_tsn

class VideoContrastive(nn.Module):

	def __init__(self) -> None:
		super().__init__()

		self.a1 = torchvision.transforms.RandomAffine(10)
		self.a2 = torchvision.transforms.RandomHorizontalFlip(p=0.5)
		self.a3 = torchvision.transforms.RandomVerticalFlip(p=0.5)

		self.tsn_loss = nn.CrossEntropyLoss()
		self.similarity = nn.MSELoss()

		#self.tsn_fc = nn.Linear(192, 4)
		self.tsn_fc = nn.Sequential(nn.Linear(96, 4))

		self.mlp_inter_decode_q = MLP(1, 32, 50, do_prob=0.5, do_bn=False)
		self.mlp_intra_decode_q = MLP(1, 32, 50, do_prob=0.5, do_bn=False)
		self.mlp_segments_decode_q = MLP(5, 32, 50, do_prob=0.5, do_bn=False)
		self.mlp_tsn_decode = MLP(5, 32, 50, do_prob=0.5, do_bn=False)

		self.mlp_inter_decode_k = MLP(1, 32, 50, do_prob=0.5, do_bn=False)
		self.mlp_intra_decode_k = MLP(1, 32, 50, do_prob=0.5, do_bn=False)
		self.mlp_segments_decode_k = MLP(5, 32, 50, do_prob=0.5, do_bn=False)

		self.mlp_inter_encode_q = nn.Sequential(
			MLP(600, 512, 300, do_prob=0.5),
			MLP(300, 256, 128, do_prob=0.5),
			MLP(128, 128, 64, do_prob=0.5)
		)

		self.mlp_intra_encode_q = nn.Sequential(
			MLP(600, 512, 300, do_prob=0.5),
			MLP(300, 256, 128, do_prob=0.5),
			MLP(128, 128, 64, do_prob=0.5)
		)

		self.mlp_segments_encode_q = nn.Sequential(
			MLP(600, 512, 300, do_prob=0.5),
			MLP(300, 256, 128, do_prob=0.5),
			MLP(128, 128, 64, do_prob=0.5)
		)

		#self.mlp_inter_encode_q = MLP(600, 128, 64, do_prob=0.5)
		#self.mlp_intra_encode_q = MLP(600, 128, 64, do_prob=0.5)
		#self.mlp_segments_encode_q = MLP(600, 128, 64, do_prob=0.5)

		#self.mlp_inter_encode_k = MLP(600, 128, 64, do_prob=0.5)
		#self.mlp_intra_encode_k = MLP(600, 128, 64, do_prob=0.5)
		#self.mlp_segments_encode_k = MLP(600, 128, 64, do_prob=0.5)

		self.mlp_inter_encode_k = nn.Sequential(
			MLP(600, 512, 300, do_prob=0.5),
			MLP(300, 256, 128, do_prob=0.5),
			MLP(128, 128, 64, do_prob=0.5)
		)

		self.mlp_intra_encode_k = nn.Sequential(
			MLP(600, 512, 300, do_prob=0.5),
			MLP(300, 256, 128, do_prob=0.5),
			MLP(128, 128, 64, do_prob=0.5)
		)

		self.mlp_segments_encode_k = nn.Sequential(
			MLP(600, 512, 300, do_prob=0.5),
			MLP(300, 256, 128, do_prob=0.5),
			MLP(128, 128, 64, do_prob=0.5)
		)

		self.mlp_tsn_encode = nn.Sequential(
			MLP(1200, 1024, 512, do_prob=0.5),
			MLP(512, 512, 256, do_prob=0.5),
			MLP(256, 128, 128, do_prob=0.5),
			MLP(128, 64, 32, do_prob=0.5)
		)

		#self.mlp_tsn_encode = MLP(1200, 128, 32, do_prob=0.5)

		self.queue_size = 33600
		self.T = 0.07
		self.k_segments = 5

		self.register_buffer("queue_inter", torch.randn(self.queue_size, 64, 3))
		self.queue_inter = F.normalize(self.queue_inter, dim=0)
		self.register_buffer("queue_inter_ptr", torch.zeros(1, dtype=torch.long))

		self.register_buffer("queue_video", torch.randn(self.queue_size, 64, 3))
		self.queue_video = F.normalize(self.queue_inter, dim=0)
		self.register_buffer("queue_video_ptr", torch.zeros(1, dtype=torch.long))

	def __aug(self, x):
		#x: 2N, 25, 1, 3
		x_aug = torch.transpose(x, 1, 3).clone().detach().cuda()
		x_aug = self.a1(x_aug)
		x_aug = self.a2(x_aug)
		x_aug = self.a3(x_aug)

		return x, torch.transpose(x_aug, 1, 3)

	def forward_distribute(self, x):
		#x.size = 2N, 25, 50 , 3
		N_2, V, T, C = x.size()
		#batch_size /= 2
		#M = 2
		#x_segments = torch.Tensor(self.k_segments, N_2, V, T//self.k_segments, C)
		seg_q = torch.Tensor(self.k_segments, N_2, V, C).cuda()
		seg_k = torch.Tensor(self.k_segments, N_2, V, C).cuda()

		with torch.no_grad():
			v1_inter = torch.unsqueeze(x[:, :, 0, :], dim=2)
			v1_inter, v1_inter_aug = self.__aug(v1_inter)
			v2_inter = torch.unsqueeze(x[:, :, 25, :], dim=2)
			v3_inter = torch.unsqueeze(x[:, :, -1, :], dim=2)

			v1_intra = torch.unsqueeze(x[:, :, 0, :], dim=2)
			v1_intra, v1_intra_aug = self.__aug(v1_intra)
			v2_intra = torch.unsqueeze(x[:, :, 25, :], dim=2)
			v3_intra = torch.unsqueeze(x[:, :, -1, :], dim=2)
			#v.shape = 2N, 25, 1, 3

			segment_size = T//self.k_segments
			for i in range(self.k_segments):
				a = x[:, :, segment_size*i:segment_size*(i+1), :]
				p = x[:, :, segment_size*i, :]
				k = x[:, :, segment_size*(i+1)-1, :]

				#x_segments[i] = a
				seg_q[i] = p
				seg_k[i] = k

		seg_q = [torch.unsqueeze(i, 2) for i in seg_q]
		seg_k = [torch.unsqueeze(i, 2) for i in seg_k]

		seg_q = torch.cat([i for i in seg_q], dim=2)
		seg_k = torch.cat([i for i in seg_k], dim=2)

		tsn_q = seg_q.clone().detach()
		tsn_k = seg_k.clone().detach()

		seg_q = torch.transpose(self.mlp_segments_decode_q(torch.transpose(seg_q, 2, 3)), 2, 3)
		seg_k = torch.transpose(self.mlp_segments_decode_q(torch.transpose(seg_k, 2, 3)), 2, 3)

		order_label = 0  # 00, 01, 10, 11
		shuffle_q = rn.randint(0, 1)
		shuffle_k = rn.randint(0, 1)

		if shuffle_q and shuffle_k:
			tsn_q = tsn_q[:, :, torch.randperm(self.k_segments), :]
			tsn_k = tsn_k[:, :, torch.randperm(self.k_segments), :]
			order_label = 3
		elif shuffle_q and not shuffle_k:
			tsn_q = tsn_q[:, :, torch.randperm(self.k_segments), :]
			order_label = 2
		elif not shuffle_q and shuffle_k:
			tsn_k =  tsn_k[:, :, torch.randperm(self.k_segments), :]
			order_label  = 1
		else:
			order_label = 0

		tsn_q = torch.transpose(self.mlp_tsn_decode(torch.transpose(tsn_q, 2, 3)), 2, 3)
		tsn_k = torch.transpose(self.mlp_tsn_decode(torch.transpose(tsn_k, 2, 3)), 2, 3)

		v1_inter = torch.transpose(self.mlp_inter_decode_q(torch.transpose(v1_inter, 2, 3)), 2, 3)
		v1_inter_aug = torch.transpose(self.mlp_inter_decode_k(torch.transpose(v1_inter_aug, 2, 3)), 2, 3)
		v2_inter = torch.transpose(self.mlp_inter_decode_k(torch.transpose(v2_inter, 2, 3)), 2, 3)
		v3_inter = torch.transpose(self.mlp_inter_decode_k(torch.transpose(v3_inter, 2, 3)), 2, 3)

		v1_intra = torch.transpose(self.mlp_intra_decode_q(torch.transpose(v1_intra, 2, 3)), 2, 3)
		v1_intra_aug = torch.transpose(self.mlp_intra_decode_k(torch.transpose(v1_intra_aug, 2, 3)), 2, 3)
		v2_intra = torch.transpose(self.mlp_intra_decode_k(torch.transpose(v2_intra, 2, 3)), 2, 3)
		v3_intra = torch.transpose(self.mlp_intra_decode_k(torch.transpose(v3_intra, 2, 3)), 2, 3)
		#v.shape = 2N, 25, 50, 3

		return v1_inter, v1_inter_aug, v2_inter, v3_inter, v1_intra, v1_intra_aug, v2_intra, v3_intra, seg_q, seg_k, tsn_q, tsn_k, order_label

	def contrast_inter(self, q, key, n=None):

		b, *u = q.size()
		l = 0
		for k in key:
			l_pos = torch.exp(self.similarity(q, k))
			l_neg = 0
			maxItem = self.queue_size//b - 1
			start = 0
			negative_pairs = []
			with torch.no_grad():
				for i in range(maxItem):
					negative_pairs.append(self.queue_inter[(start*b):((start+1)*b),:,:])
					start += 1
			for i in range(maxItem):
				l_neg += torch.exp(self.similarity(q, negative_pairs[i]))
			l += -torch.log(1e-12+(l_pos/(l_neg+l_pos)))
		return l/3

	def contrast_intra(self, q, key, n):
		l_pos = torch.exp(self.similarity(q, key))
		l_neg = 0
		for i in n:
			l_neg = torch.exp(self.similarity(q, i))
		l = -torch.log(1e-12+(l_pos/(l_neg+l_pos)))
		return l

	def contrast_video(self, q, k):
		b, *u = q.size()
		l_neg = 0
		l_pos = torch.exp(self.similarity(q, k))
		maxItem = self.queue_size//b - 1
		start = 0
		negative_pairs = []
		with torch.no_grad():
			for i in range(maxItem):
				negative_pairs.append(self.queue_video[(start*b):((start+1)*b),:,:].clone().detach())
				start += 1
		for i in range(maxItem):
			l_neg += torch.exp(self.similarity(q, negative_pairs[i]))

		l = -torch.log(1e-12+(l_pos/(l_neg+l_pos)))
		return l

	def contrast_tsn(self, q, label):

		target = torch.Tensor(q.size()[0]*[label]).cuda()
		target = target.to(torch.int64)
		q = torch.reshape(q, (q.size()[0], -1))
		query = self.tsn_fc(q)
		return self.tsn_loss(query, target)


	@torch.no_grad()
	def update_ptr(self, b):
		assert self.queue_size % b == 0
		self.queue_inter_ptr[0] = (self.queue_inter_ptr[0] + 3*b) % self.queue_size
		self.queue_video_ptr[0] = (self.queue_video_ptr[0] + b) % self.queue_size

	@torch.no_grad()
	def _dequeue_and_enqueue(self, inter, video):
		try:
			b = video.shape[0]
			ptr = int(self.queue_video_ptr)
			gpu_index = video.device.index
			self.queue_video[(ptr + b * gpu_index):(ptr+b*(gpu_index+1)), :, :] = video
		except Exception as e:
			pass

		try:
			b *= 3
			inter = list(inter)
			inter = torch.cat([inter[0], inter[1], inter[2]], dim=0)
			ptr = int(self.queue_inter_ptr)
			self.queue_video[(ptr + b * gpu_index):(ptr+b*(gpu_index+1)), :, :] = inter
		except:
			pass

