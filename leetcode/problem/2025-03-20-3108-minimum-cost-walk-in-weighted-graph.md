---
layout: leetcode-entry
title: "3108. Minimum Cost Walk in Weighted Graph"
permalink: "/leetcode/problem/2025-03-20-3108-minimum-cost-walk-in-weighted-graph/"
leetcode_ui: true
entry_slug: "2025-03-20-3108-minimum-cost-walk-in-weighted-graph"
---

[3108. Minimum Cost Walk in Weighted Graph](https://leetcode.com/problems/minimum-cost-walk-in-weighted-graph/description) hard
[blog post](https://leetcode.com/problems/minimum-cost-walk-in-weighted-graph/solutions/6558271/kotlin-rust-by-samoylenkodmitry-xfwx/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20032025-3108-minimum-cost-walk-in?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/EOePNX9rJ90)
![1.webp](/assets/leetcode_daily_images/253a6b4c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/933

#### Problem TLDR

Min AND of path weights #hard #bit_manipulation #union-find

#### Intuition

The AND operator is always decreasing the result.
We can travel to all vertices in the connected components.

#### Approach

* `x & -1 == x`
* `x & x == x`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

### Code

```kotlin

    fun minimumCost(n: Int, edges: Array<IntArray>, query: Array<IntArray>): List<Int> {
        val uf = IntArray(n) { it }; val c = IntArray(n) { -1 }
        fun f(a: Int): Int = if (a == uf[a]) a else f(uf[a]).also { uf[a] = it }
        for ((a, b, w) in edges) { c[f(a)] = c[f(a)] and w and c[f(b)]; uf[f(b)] = f(a) }
        return query.map { (a, b) -> if (f(a) == f(b)) c[f(a)] else -1 }
    }

```
```rust

    pub fn minimum_cost(n: i32, g: Vec<Vec<i32>>, q: Vec<Vec<i32>>) -> Vec<i32> {
        let mut c = vec![-1; n as usize]; let mut uf: Vec<_> = (0..c.len()).collect();
        let mut f = |uf: &mut Vec<usize>, a: usize| { while uf[a] != uf[uf[a]] { uf[a] = uf[uf[a]] }; uf[a] };
        for e in g { let (a, b, w) = (e[0] as usize, e[1] as usize, e[2]);
            let (ra, rb) = (f(&mut uf, a), f(&mut uf, b)); c[ra] &= w & c[rb]; uf[rb] = ra
        }
        q.iter().map(|q| { let (a, b) = (q[0] as usize, q[1] as usize);
            let (ra, rb) = (f(&mut uf, a), f(&mut uf, b)); if ra == rb { c[ra] } else { -1 }
        }).collect()
    }

```
```c++

    vector<int> minimumCost(int n, vector<vector<int>>& e, vector<vector<int>>& q) {
        vector<int> uf(n), c(n, -1), res; iota(begin(uf), end(uf), 0);
        auto f = [&](this const auto& f, int x) -> int { return x == uf[x] ? x : uf[x] = f(uf[x]); };
        for (auto& r: e) c[f(r[0])] &= r[2] & c[f(r[1])], uf[f(r[1])] = f(r[0]);
        for (auto& r: q) res.push_back(f(r[0]) == f(r[1]) ? c[f(r[0])] : -1);
        return res;
    }

```

