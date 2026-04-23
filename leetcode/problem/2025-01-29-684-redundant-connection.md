---
layout: leetcode-entry
title: "684. Redundant Connection"
permalink: "/leetcode/problem/2025-01-29-684-redundant-connection/"
leetcode_ui: true
entry_slug: "2025-01-29-684-redundant-connection"
---

[684. Redundant Connection](https://leetcode.com/problems/redundant-connection/description/) medium
[blog post](https://leetcode.com/problems/redundant-connection/solutions/6343420/kotlin-rust-by-samoylenkodmitry-amvd/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29012025-684-redundant-connection?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/NFlQvw7JVF4)
![1.webp](/assets/leetcode_daily_images/e308a422.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/879

#### Problem TLDR

First edge making a cycle in graph #medium #union_find

#### Intuition

Let's add edges into a Union-Find and check for the existing connection.

#### Approach

* size + 1 to simplify the code
* path shortening is almost optimal, another optimization is ranking but not worth the keystrokes

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun findRedundantConnection(g: Array<IntArray>): IntArray {
        val u = IntArray(g.size + 1) { it }
        fun f(a: Int): Int = if (a == u[a]) a else f(u[a]).also { u[a] = it }
        return g.first { (a, b) -> f(a) == f(b).also { u[u[a]] = u[b] }}
    }

```
```rust

    pub fn find_redundant_connection(edges: Vec<Vec<i32>>) -> Vec<i32> {
        let mut u: Vec<_> = (0..=edges.len()).collect();
        edges.into_iter().find(|e| { let (a, b) = (e[0] as usize, e[1] as usize);
            while u[a] != u[u[a]] { u[a] = u[u[a]]};
            while u[b] != u[u[b]] { u[b] = u[u[b]]};
            let r = u[a] == u[b]; let a = u[a]; u[a] = u[b]; r}).unwrap()
    }

```
```c++

    vector<int> findRedundantConnection(vector<vector<int>>& g) {
        int u[1001]; iota(u, u + 1001, 0);
        auto f = [&](this const auto f, int a) {
            while (u[a] != u[u[a]]) u[a] = u[u[a]]; return u[a];};
        for (auto& e: g) if (f(e[0]) == f(e[1])) return e;
            else u[u[e[0]]] = u[e[1]];
        return {};
    }

```

