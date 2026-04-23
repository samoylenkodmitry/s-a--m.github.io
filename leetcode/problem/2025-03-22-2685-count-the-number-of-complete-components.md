---
layout: leetcode-entry
title: "2685. Count the Number of Complete Components"
permalink: "/leetcode/problem/2025-03-22-2685-count-the-number-of-complete-components/"
leetcode_ui: true
entry_slug: "2025-03-22-2685-count-the-number-of-complete-components"
---

[2685. Count the Number of Complete Components](https://leetcode.com/problems/count-the-number-of-complete-components/description/) medium
[blog post](https://leetcode.com/problems/count-the-number-of-complete-components/solutions/6566250/kotlin-rust-by-samoylenkodmitry-2pur/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22032025-2685-count-the-number-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/7yDe-7IlgAk)
![1.webp](/assets/leetcode_daily_images/a24f4ea4.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/935

#### Problem TLDR

Count fully connected components #medium #union-find #graph

#### Intuition

Detect connected components with Union-Find or DFS.
Check for conditions:
* number of edges is `v * (v - 1) / 2` to vertices
* or, each vertice has `e - 1` outgoing edges

![x.png](/assets/leetcode_daily_images/7be86e65.webp)

#### Approach

* the total of `50` can speed up Rust and c++ by using primitice arrays

#### Complexity

- Time complexity:
$$O(n + e)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun countCompleteComponents(n: Int, edges: Array<IntArray>): Int {
        val uf = IntArray(n) { it }; val sz = IntArray(n)
        fun f(a: Int): Int = if (uf[a] == a) a else f(uf[a]).also { uf[a] = it }
        for ((a, b) in edges) if (f(a) == f(b)) sz[f(a)]++
            else { sz[f(a)] += 1 + sz[f(b)]; uf[f(b)] = f(a) }
        return (0..<n).groupBy(::f).count { (k, v) -> sz[k] == v.size * (v.size - 1) / 2  }
    }

```
```kotlin

    fun countCompleteComponents(n: Int, edges: Array<IntArray>): Int {
        val g = Array(n) { ArrayList<Int>() }; val ms = HashSet<Int>()
        for ((a, b) in edges) { g[a] += b; g[b] += a }
        return (0..<n).count { i ->
            val s = HashSet<Int>()
            fun dfs(i: Int) { if (s.add(i) && ms.add(i)) g[i].onEach(::dfs) }
            dfs(i); s.all { g[it].size == s.size - 1 }
        }
    }

```
```rust

    pub fn count_complete_components(n: i32, edges: Vec<Vec<i32>>) -> i32 {
        let mut e = [(0, 1); 50]; let mut uf: Vec<_> = (0..50).collect();
        let mut f = |a: usize, uf: &mut Vec<usize>| { while uf[a] != uf[uf[a]] { uf[a] = uf[uf[a]]} uf[a] };
        for x in edges { let (a, b) = (f(x[0] as usize, &mut uf), f(x[1] as usize, &mut uf));
            e[a].0 += 1; if a != b { e[a].0 += e[b].0; e[a].1 += e[b].1; e[b].1 = 0; uf[b] = a }}
        (0..n as usize).filter(|&i| e[i].1 > 0 && e[i].1 * (e[i].1 - 1) / 2 == e[i].0).count() as _
    }

```
```c++

    int countCompleteComponents(int n, vector<vector<int>>& edges) {
        int uf[50], sz[50] = {}, cnt[50] = {}, res = 0; iota(uf, uf + n, 0), fill_n(cnt, n, 1);
        auto f = [&](this const auto& f, int a) -> int { return uf[a] == a ? a : uf[a] = f(uf[a]); };
        for (auto& e : edges) if (int a = f(e[0]), b = f(e[1]); a == b) sz[a]++;
            else sz[a] += 1 + sz[b], cnt[a] += cnt[b], cnt[b] = 0, uf[b] = a;
        for (int i = 0; i < n; ++i) res += cnt[i] && cnt[i] * (cnt[i] - 1) / 2 == sz[i]; return res;
    }

```

