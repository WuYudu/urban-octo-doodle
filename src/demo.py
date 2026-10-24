"""最小演示：只要 numpy 就能跑，不需要任何求解器依赖。

    python src/demo.py

做的事：造一个 3 堆 x 3 个物体的料箱 -> 写成 QUBO -> 用贪心剥离给出一条顺序
-> 用 decode 回读 -> 检查"可行解的能量恒等于负分数"这条不变量。
"""
import numpy as np

import qubo
import scene as S
import solvers as V

M_CLUSTERS, PER_CLUSTER = 3, 3
T = 5
SEED = 0


def main() -> None:
    sc = S.make_pile_scene(
        n_clusters=M_CLUSTERS, per_cluster=PER_CLUSTER, seed=SEED, budget=T)

    print("=== 场景 ===")
    print("物体 {} 个 = {} 堆 x 每堆 {} 个，步数预算 T = {}".format(
        sc.n_obj, M_CLUSTERS, PER_CLUSTER, T))
    print("阻塞关系（谁必须先在下面那个之前被移走）：")
    for o in range(sc.n_obj):
        bl = sorted(sc.blocked[o])
        print("  obj{:>2}  lv={}  score={:.3f}  需要先移走 {}".format(
            o, sc.levels[o], sc.score(o), bl if bl else "无"))

    pack = qubo.build(sc, T)
    print("\n=== QUBO ===")
    print("变量 n = M x (T+1) = {} x {} = {}".format(sc.n_obj, T + 1, pack.n))
    print("非零项 {} 个，罚权重 lam = {:.1f}，常数项 offset = {:.1f}".format(
        pack.nnz, pack.lam, pack.offset))

    res = V.greedy(sc, T)
    print("\n=== 贪心剥离 ===")
    print("分数 {:.4f}，可行 {}，耗时 {:.0f} us".format(
        res.score, res.feasible, res.time_ms * 1e3))
    removed = set()
    for step, o in enumerate(res.seq, 1):
        why = "（本步可用）" if not (sc.blocked[o] - removed) else "（不该出现）"
        print("  第{}步  obj{:>2}  score={:.3f}  {}".format(step, o, sc.score(o), why))
        removed.add(o)

    S_ = T + 1
    x = np.zeros(pack.n, dtype=np.int64)
    for step, o in enumerate(res.seq):
        x[qubo.vid(o, step, S_)] = 1
    for o in range(sc.n_obj):
        if o not in res.seq:
            x[qubo.vid(o, T, S_)] = 1

    e = pack.energy(x)
    print("\n=== 不变量检查 ===")
    print("这条序列在 QUBO 里的能量 E = {:.6f}".format(e))
    print("负的总分数           -S = {:.6f}".format(-res.score))
    print("|E + S| = {:.3e}  ->  {}".format(
        abs(e + res.score), "恒等式成立" if abs(e + res.score) < 1e-9 else "不成立！"))

    back = qubo.decode(sc, x, T)
    print("decode 回读：可行 {}，分数 {:.4f}，序列 {}".format(
        back["feasible"], back["score"], back["seq"]))


if __name__ == "__main__":
    main()
