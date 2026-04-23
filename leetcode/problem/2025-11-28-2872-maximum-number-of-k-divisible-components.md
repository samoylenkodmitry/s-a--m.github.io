---
layout: leetcode-entry
title: "2872. Maximum Number of K-Divisible Components"
permalink: "/leetcode/problem/2025-11-28-2872-maximum-number-of-k-divisible-components/"
leetcode_ui: true
entry_slug: "2025-11-28-2872-maximum-number-of-k-divisible-components"
---

[2872. Maximum Number of K-Divisible Components](https://leetcode.com/problems/maximum-number-of-k-divisible-components/description/) hard
[blog post](https://leetcode.com/problems/maximum-number-of-k-divisible-components/solutions/7378983/kotlin-rust-by-samoylenkodmitry-8dlc/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28112025-2872-maximum-number-of-k?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/VI4osIc9PQo)

![beaee647-7205-4da8-b1a9-718568cec3f4 (1).webp](/assets/leetcode_daily_images/d65f73ea.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1187

#### Problem TLDR

Max cuts where every subtree sum %k #hard #dfs

#### Intuition

```j
    // any subtree sum %k is a valid cut
```

#### Approach

* DFS: start from any, return sum, track %k each
* BFS: start from leafs, accumulate sums in v, track %k each

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 34ms
    fun maxKDivisibleComponents(n: Int, e: Array<IntArray>, v: IntArray, k: Int): Int {
        val g = Array(n) { ArrayList<Int>() }; for ((a,b) in e) { g[a] += b; g[b] += a }
        fun dfs(i: Int, p: Int): Int =
            (g[i].sumOf { j -> if (j==p) 0 else dfs(j, i)} + if (v[i]%k<1)1 else 0)
            .also { v[p] += v[i]%k }
        return dfs(0, 0)
    }
```
```rust
// 32ms
    pub fn max_k_divisible_components(n: i32, e: Vec<Vec<i32>>, mut v: Vec<i32>, k: i32) -> i32 {
        let (mut g, mut d) = (vec![Vec::new(); v.len()], vec![0; v.len()]);
        for p in e {let (a,b) = (p[0]as usize,p[1]as usize);g[a].push(b);g[b].push(a);d[a]+=1;d[b]+=1}
        let mut q = VecDeque::from_iter((0..v.len()).filter(|&i|d[i]<2)); let mut r = 0;
        while let Some(u) = q.pop_front() { if v[u] % k < 1 { r += 1 }
            for &x in &g[u] { if d[x] > 0 { v[x] += v[u]%k; d[x] -=1; if d[x]==1 { q.push_back(x);}}}
        }; r
    }
```

