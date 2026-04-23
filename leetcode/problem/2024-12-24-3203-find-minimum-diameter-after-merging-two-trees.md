---
layout: leetcode-entry
title: "3203. Find Minimum Diameter After Merging Two Trees"
permalink: "/leetcode/problem/2024-12-24-3203-find-minimum-diameter-after-merging-two-trees/"
leetcode_ui: true
entry_slug: "2024-12-24-3203-find-minimum-diameter-after-merging-two-trees"
---

[3203. Find Minimum Diameter After Merging Two Trees](https://leetcode.com/problems/find-minimum-diameter-after-merging-two-trees/description/) hard
[blog post](https://leetcode.com/problems/find-minimum-diameter-after-merging-two-trees/solutions/6180735/kotlin-rust-by-samoylenkodmitry-6638/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23122024-3203-find-minimum-diameter?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/oaM78yNs1BY)
[deep-dive](https://notebooklm.google.com/notebook/06ca0e0a-e5c5-43f3-b729-4386f3c2124c/audio)
![1.webp](/assets/leetcode_daily_images/add17f4d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/842

#### Problem TLDR

Diameter of 2 connected trees #hard #graph #toposort

#### Intuition

Can't solve without hint.
The hint: 1. connect by centers 2. center of tree is on the diameter 3. diameter if a two-bfs ends connected

There is another approach to find the diameter: topological sort.

#### Approach

* in toposort, the clever way to find a diameter: `dep[j] + dep[i] + 1` is a total length for both ends connected by one edge

#### Complexity

- Time complexity:
$$O(EV)$$

- Space complexity:
$$O(EV)$$

#### Code

```kotlin

    fun minimumDiameterAfterMerge(edges1: Array<IntArray>, edges2: Array<IntArray>): Int {
        val g = List(2) { HashMap<Int, ArrayList<Int>>() }; val q = ArrayDeque<Int>()
        for ((g, e) in g.zip(listOf(edges1, edges2))) for ((a, b) in e) {
            g.getOrPut(a) { arrayListOf() } += b; g.getOrPut(b) { arrayListOf() } += a }
        fun bfs(s: Int, g: Map<Int, List<Int>>): List<Int> {
            q.clear(); q += s; val seen = IntArray(g.size + 1); seen[s] = 1; var d = 0; var l = s
            while (q.size > 0) {
                val c = q.removeFirst(); l = c;
                for (n in g[c] ?: listOf())
                  if (seen[n] == 0) { seen[n] = seen[c] + 1; q += n; d = max(d, seen[n]) }
            }
            return listOf(l, d - 1)
        }
        val d1 = bfs(bfs(0, g[0])[0], g[0])[1]; val d2 = bfs(bfs(0, g[1])[0], g[1])[1]
        return maxOf(d1, d2, (d1 + 1) / 2 + (d2 + 1) / 2 + 1)
    }

```
```rust

    pub fn minimum_diameter_after_merge(edges1: Vec<Vec<i32>>, edges2: Vec<Vec<i32>>) -> i32 {
        let mut g = [HashMap::new(), HashMap::new()];
        for (g, e) in g.iter_mut().zip([edges1, edges2]) { for e in e {
            for ix in 0..2 { g.entry(e[ix % 2]).or_insert(vec![]).push(e[(ix + 1) % 2]); }
        } }
        fn bfs(s: i32, g: &HashMap<i32, Vec<i32>>) -> (i32, i32) {
            let mut q = VecDeque::from([s]); let mut seen = vec![0; g.len() + 1];
            seen[s as usize] = 1; let (mut l, mut d) = (s, 0);
            while let Some(c) = q.pop_front() { l = c; if let Some(sibl) = g.get(&c) {
                for &n in sibl { if seen[n as usize] == 0 {
                    seen[n as usize] = seen[c as usize] + 1;
                    d = d.max(seen[n as usize]);
                    q.push_back(n);
            }}}}
            (l, d - 1)
        }
        let d1 = bfs(bfs(0, &g[0]).0, &g[0]).1; let d2 = bfs(bfs(0, &g[1]).0, &g[1]).1;
        d1.max(d2).max(1 + (d1 + 1) / 2 + (d2 + 1) / 2)
    }

```
```c++

    int minimumDiameterAfterMerge(vector<vector<int>>& e1, vector<vector<int>>& e2) {
        auto f = [](this auto const& f, vector<vector<int>>& e) -> int {
            int n = e.size() + 1, res = 0; queue<int> q;
            vector<vector<int>> g(n); vector<int> deg(n), dep(n), vis(n);
            for (const auto &e: e) {
                g[e[0]].push_back(e[1]);
                g[e[1]].push_back(e[0]);
            }
            for (int i = 0; i < n; ++i) if ((deg[i] = g[i].size()) == 1) q.push(i);
            while (q.size()) {
                int i = q.front(); q.pop(); vis[i] = 1;
                for (int j: g[i]) {
                    if (--deg[j] == 1) q.push(j);
                    if (!vis[j]) {
                        res = max(res, dep[j] + dep[i] + 1);
                        dep[j] = max(dep[j], dep[i] + 1);
                    }
                }
            }
            return res;
        };
        int d1 = f(e1), d2 = f(e2);
        return max({d1, d2, (d1 + 1) / 2 + (d2 + 1) / 2 + 1});
    }

```

