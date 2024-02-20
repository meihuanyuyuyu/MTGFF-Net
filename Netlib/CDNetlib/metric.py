from sklearn.metrics import f1_score, mean_squared_error


def F1(pred, target):
    r"input Tensor"
    N = len(pred)
    f1_s = []
    for _ in range(N):
        p = pred[_].cpu().numpy()
        t = target[_].cpu().numpy()
        p = p.reshape(-1)
        t = t.reshape(-1)
        f1_s.append(f1_score(t, p, average="macro"))
    return sum(f1_s) / N


def mse(pred, target):
    r"input Tensor"
    N = len(pred)
    mse = []
    for _ in range(N):
        p = pred[_].cpu().numpy()
        t = target[_].cpu().numpy()
        p = p.reshape(-1)
        t = t.reshape(-1)
        mse.append(mean_squared_error(t, p))
    return sum(mse) / N
