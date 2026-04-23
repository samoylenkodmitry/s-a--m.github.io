---
layout: leetcode-entry
title: "1857. Largest Color Value in a Directed Graph"
permalink: "/leetcode/problem/2025-05-26-1857-largest-color-value-in-a-directed-graph/"
leetcode_ui: true
entry_slug: "2025-05-26-1857-largest-color-value-in-a-directed-graph"
---

[1857. Largest Color Value in a Directed Graph](https://leetcode.com/problems/largest-color-value-in-a-directed-graph/description/) hard
[blog post](https://leetcode.com/problems/largest-color-value-in-a-directed-graph/solutions/6782066/kotlin-rust-by-samoylenkodmitry-ould/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26052025-1857-largest-color-value?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/YS4FROIOzsQ)
![1.webp](/assets/leetcode_daily_images/7ca8e076.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1000

#### Problem TLDR

Max color freq path #hard #dp #toposort

#### Intuition

```j
    // for every node
    // (freq of colors)
    // we want max(freq)
```

As graph is `directed asyclic graph`, we can assume each node has definite answer of top color frequencies of all path from it.
Walk those paths in DFS or BFS with toposort.

#### Approach

* to check cycle use a hashset, don't forget to remove from it after we done, as we can have two different valid paths to the same node
* the toposort is: 1) count incoming nodes 2) always add to the queue `zero` incoming nodes

#### Complexity

- Time complexity:
$$O(VE)$$

- Space complexity:
$$O(V + E)$$

#### Code

```kotlin

// 1213ms
    fun largestPathValue(c: String, e: Array<IntArray>): Int {
        val g = Array(c.length) { ArrayList<Int>() }; for ((a, b) in e) g[a] += b
        val v = HashSet<Int>(); val dp = HashMap<Int, IntArray>()
        fun dfs(i: Int): IntArray? = dp.getOrPut(i) { val res = IntArray(26)
            for (s in g[i]) {
                if (!v.add(s)) return@dfs null; val next = dfs(s) ?: return@dfs null
                v.remove(s); for (j in 0..25) res[j] = max(res[j], next[j])
            }; res[c[i] - 'a']++; res
        }
        return c.indices.maxOf { dfs(it)?.max() ?: return -1 }
    }

```
```kotlin

// 802ms
    fun largestPathValue(c: String, e: Array<IntArray>): Int {
        val d = IntArray(c.length); val g = Array(d.size) { ArrayList<Int>() }
        val cnt = Array(d.size) { IntArray(26) }; for ((a, b) in e) { g[a] += b; ++d[b] }
        var q = ArrayList<Int>(); var q1 = ArrayList<Int>(); var r = 0; var v = 0;
        for (i in d.indices) if (d[i] == 0) q += i;
        while (q.size > 0) {
            for (i in q) {
                ++v; r = max(r, ++cnt[i][c[i] - 'a'])
                for (j in g[i]) {
                    for (k in 0..25) cnt[j][k] = max(cnt[j][k], cnt[i][k])
                    if (--d[j] == 0) q1 += j
                }
            }
            q = q1.also { q1 = q; q1.clear() }
        }
        return if (v == d.size) r else -1
    }

```
```rust

// 69ms
    pub fn largest_path_value(c: String, e: Vec<Vec<i32>>) -> i32 {
        let (mut g, c, mut d, mut q) = (vec![vec![]; c.len()], c.as_bytes(), vec![0; c.len()], vec![]);
        let (mut cnt, mut q1, mut r, mut v) = (vec![vec![0; 26]; c.len()], vec![], 0, 0);
        for e in e  { let (a, b) = (e[0] as usize, e[1] as usize); g[a].push(b); d[b] += 1; }
        for i in 0..d.len() { if d[i] == 0 { q.push(i) }}
        while q.len() > 0 {
            for &i in &q { let c = (c[i] - b'a') as usize;
                v += 1; cnt[i][c] += 1; r = r.max(cnt[i][c]);
                for &j in &g[i] {
                    for k in 0..26 { cnt[j][k] = cnt[j][k].max(cnt[i][k]); }
                    d[j] -= 1; if d[j] == 0 { q1.push(j) } }
            }  (q, q1) = (q1, q); q1.clear()
        } if v == d.len() { r } else { -1 }
    }

```
```c++

// 312ms
    int largestPathValue(string c, vector<vector<int>>& e) {
        int n = size(c), v = 0, r = 0; vector<array<int,26>> cnt(n);
        vector<vector<int>> g(n); vector<int> d(n); queue<int> q;
        for (auto& p : e) { g[p[0]].push_back(p[1]); ++d[p[1]]; }
        for (int i = 0; i < n; i++) if (!d[i]) q.push(i);
        while (!q.empty()) {
            int i = q.front(); q.pop(); ++v; r = max(r, ++cnt[i][c[i] - 'a']);
            for (int j: g[i]) {
                for (int k = 0; k < 26; k++) cnt[j][k] = max(cnt[j][k], cnt[i][k]);
                if (--d[j] == 0) q.push(j);
            }
        }
        return v == n ? r : -1;
    }

 ```

