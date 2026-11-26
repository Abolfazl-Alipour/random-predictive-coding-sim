import math
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

# ---------- data: random tokens with planted motifs ----------
def make_data(n_seq=200, length=24, K=16, motifs=((1, 2, 3), (5, 6, 7), (9, 10, 11))):
    data = []
    for _ in range(n_seq):
        seq = np.random.randint(0, K, size=length).tolist()
        for motif in motifs:
            pos = np.random.randint(0, length - len(motif))
            seq[pos:pos + len(motif)] = list(motif)
        data.append(seq)
    return data

# ---------- fixed random reservoir module ----------
class ReservoirModule(torch.nn.Module):
    def __init__(self, d_in, n_exp=256, n_res=200, n_out=64,
                 k_e=32, alpha=0.3, rho=0.9, learn_prox=False):
        super().__init__()
        self.n_res, self.n_out = n_res, n_out
        self.k_e, self.alpha = k_e, alpha
        self.learn_prox = learn_prox
        W_exp = torch.randn(n_exp, d_in) / math.sqrt(d_in)
        self.register_buffer("W_exp", W_exp)
        W_in = torch.randn(n_res, n_exp) / math.sqrt(n_exp)
        self.register_buffer("W_in", W_in)
        W_res = torch.randn(n_res, n_res) * 0.1
        ev = torch.linalg.eigvals(W_res).abs().max()
        W_res = W_res / ev * rho
        self.register_buffer("W_res", W_res)
        W_prox = torch.randn(n_out, n_res) / math.sqrt(n_res) * 0.5
        self.register_buffer("W_prox", W_prox)
        self.register_buffer("r", torch.zeros(n_res))
        self.register_buffer("inhib", torch.zeros(n_res))

    def reset(self):
        self.r.zero_()
        self.inhib.zero_()

    def forward(self, u):
        e = torch.relu(self.W_exp @ u)
        top = torch.topk(e, self.k_e).indices
        e_sparse = torch.zeros_like(e)
        e_sparse[top] = e[top]
        r_new = (1 - self.alpha) * self.r + self.alpha * torch.tanh(
            self.W_in @ e_sparse + self.W_res @ self.r - self.inhib)
        self.r = r_new
        self.inhib.zero_()
        p = self.W_prox @ self.r
        return p, self.r, e_sparse

# ---------- two-module RPC network ----------
class RPCNetwork(torch.nn.Module):
    def __init__(self, K=16, n_exp=256, n_res=200, n_out=64,
                 theta_p=0.5, theta_d=0.5, theta_f=0.5,
                 eta_b=0.05, eta_c=0.05, w_max=2.0, s_inh=0.5, g2=1.0):
        super().__init__()
        self.K = K
        self.mod1 = ReservoirModule(K, n_exp, n_res, n_out)
        self.mod2 = ReservoirModule(n_res, n_exp, n_res, n_out)
        self.theta_p, self.theta_d, self.theta_f = theta_p, theta_d, theta_f
        self.eta_b, self.eta_c, self.w_max = eta_b, eta_c, w_max
        self.s_inh, self.g2 = s_inh, g2
        self.register_buffer("W_12", torch.randn(n_res, n_res) / math.sqrt(n_res))
        self.register_buffer("W_dist1", torch.randn(n_out, n_out) * 0.05)
        self.register_buffer("W_conf", torch.randn(n_out, n_out) / math.sqrt(n_out))
        self.register_buffer("y2_prev", torch.zeros(n_out))
        self.register_buffer("r2_prev", torch.zeros(n_res))
        self.register_buffer("p2_prev", torch.zeros(n_out))
        self.prototypes = None

    def reset(self):
        self.mod1.reset()
        self.mod2.reset()
        self.y2_prev.zero_()
        self.r2_prev.zero_()
        self.p2_prev.zero_()

    def build_prototypes(self, data):
        sums = torch.zeros(self.K, self.mod1.n_out)
        counts = torch.zeros(self.K)
        self.reset()
        with torch.no_grad():
            for seq in data[:20]:
                self.reset()
                for tok in seq:
                    x = torch.nn.functional.one_hot(
                        torch.tensor(tok), self.K).float()
                    p1, _, _ = self.mod1.forward(x)
                    sums[tok] += p1
                    counts[tok] += 1
        self.prototypes = sums / counts[:, None].clamp(min=1)
        self.reset()

    def predict_from_distal(self, d1):
        if self.prototypes is None:
            return -1
        a = torch.nn.functional.normalize(d1[None, :], dim=1)
        b = torch.nn.functional.normalize(self.prototypes, dim=1)
        return int(torch.argmax(a @ b.T).item())

    def step(self, tok, train=True, use_confirmation=True, use_burst=True):
        x = torch.nn.functional.one_hot(torch.tensor(tok), self.K).float()
        with torch.no_grad():
            d1 = self.W_dist1 @ self.y2_prev
            p1, r1, _ = self.mod1.forward(x)
            u2 = torch.tanh(self.W_12 @ r1)
            p2, r2, _ = self.mod2.forward(u2)
            y2 = (self.g2 * p2 > self.theta_f).float()
            if use_burst:
                b1 = ((p1 > self.theta_p) & (d1 > self.theta_d)).float()
            else:
                b1 = torch.sigmoid(p1 * d1)
            pred = self.predict_from_distal(d1)
            if train:
                self.W_dist1 += self.eta_b * torch.outer(b1, self.y2_prev)
                self.W_dist1.clamp_(-self.w_max, self.w_max)
                if use_confirmation:
                    d2 = self.W_conf @ b1
                    b2 = ((self.p2_prev > self.theta_p) &
                          (d2 > self.theta_d)).float()
                    self.mod2.W_prox += self.eta_c * torch.outer(b2, self.r2_prev)
                    self.mod2.W_prox.clamp_(-self.w_max, self.w_max)
                raw = self.mod1.W_prox.T @ b1
                self.mod1.inhib = self.s_inh * raw.clamp(0, 0.8)
            self.y2_prev = y2.clone()
            self.r2_prev = r2.clone()
            self.p2_prev = p2.clone()
        return {"pred": pred, "burst_rate": float(b1.mean()),
                "y2_rate": float(y2.mean())}

def run(train_data, test_data, epochs=5, **kw):
    step_kw = {k: kw.pop(k) for k in ("use_confirmation", "use_burst") if k in kw}
    net = RPCNetwork(**kw)
    net.build_prototypes(train_data)
    for ep in range(epochs):
        for seq in train_data:
            net.reset()
            for tok in seq:
                net.step(tok, train=True, **step_kw)
    correct, total, bursts = 0, 0, []
    for seq in test_data:
        net.reset()
        for i, tok in enumerate(seq):
            out = net.step(tok, train=False)
            if i > 0:
                total += 1
                correct += int(out["pred"] == tok)
            bursts.append(out["burst_rate"])
    print(f"acc={correct / max(total, 1):.3f} chance={1 / net.K:.3f} "
          f"mean_burst={np.mean(bursts):.3f}")
    return net

if __name__ == "__main__":
    train = make_data(n_seq=200)
    test = make_data(n_seq=50)
    print("RPC full:")
    run(train, test)
    print("RPC no confirmation:")
    run(train, test, use_confirmation=False)
    print("RPC no burst gate:")
    run(train, test, use_burst=False)
