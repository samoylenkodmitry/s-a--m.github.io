---
layout: leetcode-entry
title: "1971. Find if Path Exists in Graph"
permalink: "/leetcode/problem/2024-04-21-1971-find-if-path-exists-in-graph/"
leetcode_ui: true
entry_slug: "2024-04-21-1971-find-if-path-exists-in-graph"
---

[1971. Find if Path Exists in Graph](https://leetcode.com/problems/find-if-path-exists-in-graph/description/) easy
[blog post](https://leetcode.com/problems/find-if-path-exists-in-graph/solutions/5053142/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21042024-1971-find-if-path-exists?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ZJ1P4GxHBEA)
![2024-04-21_08-22.webp](/assets/leetcode_daily_images/486f9147.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/578

#### Problem TLDR

Are `source` and `destination` connected in graph? #easy

#### Intuition

Let's check connected components with Union-Find data structure https://en.wikipedia.org/wiki/Disjoint-set_data_structure

#### Approach

We can use a HashMap or just simple array. To optimize Union-Find `root` function, we can use `path compression` step. There are other tricks (https://arxiv.org/pdf/1911.06347.pdf), but let's keep code shorter.

#### Complexity

- Time complexity:
$$O(E + V)$$, V = n, E = edges.size, assuming `root` is constant for `inverse Ackermann` function (https://codeforces.com/blog/entry/98275) (however only with all the tricks implemented, like ranks and path compressing https://cp-algorithms.com/data_structures/disjoint_set_union.html)

- Space complexity:
$$O(V)$$

#### Code

```kotlin

    fun validPath(n: Int, edges: Array<IntArray>, source: Int, destination: Int): Boolean {
        val uf = IntArray(n) { it }
        fun root(a: Int): Int { var x = a; while (x != uf[x]) x = uf[x]; uf[a] = x; return x }
        for ((a, b) in edges) uf[root(a)] = root(b)
        return root(source) == root(destination)
    }

```
```rust

    pub fn valid_path(n: i32, edges: Vec<Vec<i32>>, source: i32, destination: i32) -> bool {
        let mut uf = (0..n as usize).collect();
        fn root(uf: &mut Vec<usize>, a: i32) -> usize {
            let mut x = a as usize; while x != uf[x] { x = uf[x] }; uf[a as usize] = x; x
        }
        for ab in edges { let a = root(&mut uf, ab[0]); uf[a] = root(&mut uf, ab[1]) }
        root(&mut uf, source) == root(&mut uf, destination)
    }

```

