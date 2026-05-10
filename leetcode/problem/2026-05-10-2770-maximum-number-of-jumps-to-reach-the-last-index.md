---
layout: leetcode-entry
title: "2770. Maximum Number of Jumps to Reach the Last Index"
permalink: "/leetcode/problem/2026-05-10-2770-maximum-number-of-jumps-to-reach-the-last-index/"
leetcode_ui: true
entry_slug: "2026-05-10-2770-maximum-number-of-jumps-to-reach-the-last-index"
---

[2770. Maximum Number of Jumps to Reach the Last Index](https://leetcode.com/problems/maximum-number-of-jumps-to-reach-the-last-index/solutions/8182392/kotlin-rust-by-samoylenkodmitry-bmwi/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10052026-2770-maximum-number-of-jumps?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/EHdxxONUzGw)

https://dmitrysamoylenko.com/leetcode/

![10.05.2026.webp](/assets/leetcode_daily_images/10.05.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1355

#### Problem TLDR

Max steps to reach end by jumps n[j]-n[i] in -t..t

#### Intuition

Top-down DP: at each position i with previous position j choose to skip (dfs(i+1,p)) or take(1+dfs(i+1,i))
Bottom-up DP: for each index i if it is reachable update all next position if they reachable
Segment tree solution: for each value X query the range X-t..X+t for maximum reachable steps in this range

#### Approach

* segment tree query: if left is the right child & if right is the left child; move them sideways & update query result

#### Complexity

- Time complexity:
$$O(n^2|nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun maximumJumps(n: IntArray, t: Int) = IntArray(n.size).also { d ->
        d[0] = 1
        for (i in d.indices) if (d[i] > 0) for (j in i+1..<d.size)
            if (abs(n[i]-n[j]) <= t) d[j] = max(d[j], 1 + d[i])
    }.last() - 1
```
```rust
    pub fn maximum_jumps(n: Vec<i32>, t: i32) -> i32 {
        let mut v = n.clone(); v.sort(); v.dedup();   let m = v.len(); let mut tr = vec![-1; m*2];
        let (f,g) = (|x|v.partition_point(|&y|y<x), |x|v.partition_point(|&y|y<=x));
        (0..).zip(n).fold(0, |q, (i,x)| {
            let (mut l,mut r) = (f(x.saturating_sub(t)), g(x.saturating_add(t)).saturating_sub(1));
            let mut q = (i==0) as i32-1; l += m; r += m;
            while l <= r { if l&1 > 0 { q = q.max(tr[l]); l += 1 };  if r&1 < 1 { q = q.max(tr[r]); r -= 1 }
                l /= 2; r /= 2
            }
            if q >= 0 { q += (i>0)as i32; let mut p = f(x)+m; tr[p] = tr[p].max(q);
                while p > 1 { p /=2; tr[p] = tr[p*2].max(tr[p*2+1])}
            }; q
        })
    }
```

