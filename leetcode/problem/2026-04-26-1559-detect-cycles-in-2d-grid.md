---
layout: leetcode-entry
title: "1559. Detect Cycles in 2D Grid"
permalink: "/leetcode/problem/2026-04-26-1559-detect-cycles-in-2d-grid/"
leetcode_ui: true
entry_slug: "2026-04-26-1559-detect-cycles-in-2d-grid"
---

[1559. Detect Cycles in 2D Grid](https://leetcode.com/problems/detect-cycles-in-2d-grid/solutions/8098956/kotlin-rust-by-samoylenkodmitry-uqrc/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26042026-1559-detect-cycles-in-2d?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ebJFVQ5XPnk)

https://dmitrysamoylenko.com/leetcode/

![26.04.2026.webp](/assets/leetcode_daily_images/26.04.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1340

#### Problem TLDR

Check cycle in 2D grid

#### Intuition

The trivial idea: start DFS from each position, mark visited cells, check neighbours, skip parent.
Have to mark visited cells again with different mark to not do DFS on the same group later.
The mark in post-order should be different to make a wall for the next group.

The little bit more interesting idea: use Union-Find. Walk row by row and connect items.
If there is a cycle then right cell would be already connected.

#### Approach

* the G is for 'Good'
* use path compression for Union-Find
* do you know why we don't searching the root for the down cell in Rust?

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(nm)$$

#### Code

```kotlin
    fun containsCycle(g: Array<CharArray>) = IntArray(g.size*g[0].size){it}
        .let { u -> u.indices.any { i -> val w = g[0].size
            fun f(x: Int): Int = if (u[x]==x) x else {u[x]=f(u[x]);u[x]}
            fun c(j: Int, G: Boolean) = G && g[j/w][j%w] == g[i/w][i%w]
                &&(f(i) == f(j) || { u[f(i)] = f(j); !G}())
            c(i+1, i%w<w-1) || c(i+w, i+w<u.size)
        }}
```
```rust
    pub fn contains_cycle(g: Vec<Vec<char>>) -> bool {
        let w=g[0].len(); let z=g.len()*w; let mut u:Vec<_>=(0..z).collect();
        (0..z).any(|i|{
            if i+w<z && g[i/w+1][i%w] == g[i/w][i%w] { u[i+w] = i }
            i%w<w-1 && g[i/w][i%w+1] == g[i/w][i%w] && {
                let [a,b] = [i, i+1].map(|mut x|{while x!=u[x]{u[x]=u[u[x]];x=u[x]}x});
                a == b || {u[a] = b; 0>0}
            }
        })
    }
```
