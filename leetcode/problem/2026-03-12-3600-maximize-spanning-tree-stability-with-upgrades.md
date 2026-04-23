---
layout: leetcode-entry
title: "3600. Maximize Spanning Tree Stability with Upgrades"
permalink: "/leetcode/problem/2026-03-12-3600-maximize-spanning-tree-stability-with-upgrades/"
leetcode_ui: true
entry_slug: "2026-03-12-3600-maximize-spanning-tree-stability-with-upgrades"
---

[3600. Maximize Spanning Tree Stability with Upgrades](https://open.substack.com/pub/dmitriisamoilenko/p/12032026-3600-maximize-spanning-tree?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) hard
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/12032026-3600-maximize-spanning-tree?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12032026-3600-maximize-spanning-tree?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/uDg48-HtW5A)

![b7368096-8e84-4e37-ba9f-0e28359d9712 (1).webp](/assets/leetcode_daily_images/36211855.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1295

#### Problem TLDR

Max of min values in span tree of graph, 2*s at most k #hard #graph

#### Intuition

```j
    // i will spend no more than 30 minutes to this until i use hints
    // the must-edges should be included
    // check for cycles
    // upgrade makes sense to always do *2 (k times)
    // the minimum is min(must, notmust_k*2)
    // the question is to remove min-value edges
    // the new edges can only worsen the situation of min(must)
    // so take as few as possible
    // now the question: how to build a spanning tree with priority?
    //
    // union-find?
    //
    // now how to take all necessary edges from others to make span tree?
    // we must visit all nodes
    // take largest s first; can be not optimal
    // necessary node can be not largest
    // how to spend k optimally?
    // and to not form cycles
    //
    // 20 minute
    //
    // go from unvisited nodes?
    //
    // a-b c  (ac=1, bc=2) k=1 (ab is must, min=0)
    //
    // honestly not sure
    // both ac and bc are promising, both can make a cycle
    //
    // ok google how to build a span tree lol
    // 27 minute
    // let's just blindly sort by value and take largest
    // wrong answer, didn't work 30 minute, use hints
    // the missing part: binary search
    //
    // 1:03 another edge case
    //
```
Sort by bigger s first.
Binary Search: freeze the allowed s lvl, try to greedily add everything
Heap: just greedily add everything, then 2*s of k smallest added

#### Approach

* heap is not needed, we have a sorted order
* Union-Find to check cycles/added
* we can do a single pass if we sort by m first

#### Complexity

- Time complexity:
$$O(n+eloge)$$

- Space complexity:
$$O(e+n)$$

#### Code

```kotlin
// 238ms
    fun maxStability(n: Int, e: Array<IntArray>, k: Int): Int {
        e.sortWith(compareBy({-it[3]}, {-it[2]}))
        val u = IntArray(n) { it }; val g = ArrayList<Int>(); g += Int.MAX_VALUE
        fun f(x: Int): Int = if (u[x]==x)x else f(u[x]).also {u[x]=it}
        for ((a,b,s,m) in e) if (f(a)!=f(b)) u[f(a)]=f(b).also{if(m>0) g[0]=min(g[0],s) else g+=s} else if(m>0) return -1
        return if ((0..<n).all {f(it) == f(0)}) minOf(g[0], g[max(0,g.size-k-1)], g.last()*2) else -1
    }
```
```rust
// 33ms
    pub fn max_stability(n: i32, mut e: Vec<Vec<i32>>, k: i32) -> i32 {
        e.sort_unstable_by_key(|v| (-v[3], -v[2]));
        let mut u: Vec<_> = (0..n as usize).collect(); let (mut c, mut g) = (n, vec![i32::MAX]);
        fn f(u: &mut [usize], mut x: usize) -> usize { if x == u[x] { x } else { let a = f(u, u[x]); u[x] = a; a }}
        for v in e {
            let (a, b, s, m) = (f(&mut u, v[0] as _), f(&mut u, v[1] as _), v[2], v[3]);
            if a != b { u[a] = b; c -= 1; if m > 0 { g[0] = g[0].min(s) } else { g.push(s) } }
            else if m > 0 { return -1 }
        }
        if c > 1 { -1 } else { g[0].min(g[g.len().saturating_sub((k + 1) as usize)]).min(g[g.len() - 1]*2) }
    }
```

